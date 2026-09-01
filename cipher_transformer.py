#!/usr/bin/env python3
"""
Cipher Transformer -- from-scratch Encoder-Decoder Transformer + BLT ablation
study (ANLP A1). Auto-generated single-file build of src/ -- DO NOT hand-edit;
edit the files under src/ and re-run build_single_script.py instead.

Translates brown_plain.txt (English) into brown_cipher.txt (a per-character,
context-dependent 8-bit binary cipher) under five architectural ablations:
C1 (base: sinusoidal PE + MHA + LayerNorm + BPE), C2 (RoPE), C3 (GQA),
C4 (RMSNorm), C5 (BLT / token-free). No nn.Transformer, no
nn.MultiheadAttention -- every building block is implemented from scratch.

Requirements:
    pip install torch tokenizers huggingface_hub wandb rapidfuzz rouge-score nltk numpy pandas matplotlib

Usage:
    python cipher_transformer.py --config C1_base --epochs 40
    python cipher_transformer.py --config all --epochs 40 --push-to-hub
    python cipher_transformer.py --config C1_base --epochs 1 --wandb-mode disabled   # quick smoke test

Needs brown_plain.txt / brown_cipher.txt alongside it (or pass
--plain-path/--cipher-path). Run `python cipher_transformer.py --help` for
every flag.

Running on a cluster with no internet on compute nodes (e.g. IIIT-H's Ada):
use `--wandb-mode offline` (writes local run files under ./wandb/, no network
needed) and skip `--push-to-hub` during the job; `wandb sync` and the
Hugging Face upload (`--push-to-hub`, needs HF_TOKEN) can both be run
afterwards from the login node, pointed at the same --output-dir.
"""
import argparse
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing
from rapidfuzz.distance import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ==============================================================================
# Positional encodings: Sinusoidal (absolute) + RoPE
# (from src/models/positional.py)
# ==============================================================================

"""Positional encoding schemes: Sinusoidal (absolute) and RoPE (rotary, relative)."""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learnable) absolute positional encoding, added to the input
    embeddings: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)."""

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, D)

    def forward(self, x):
        L = x.size(1)
        if L > self.pe.size(1):
            raise ValueError(f"sequence length {L} exceeds positional encoding max_len {self.pe.size(1)}")
        return x + self.pe[:, :L, :].to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE: rotates Q/K pairs as a function of absolute position so that the
    dot product Q_m . K_n only depends on (m - n). Applied inside self-attention
    only (see attention.py) -- never cross-attention, since source and target
    occupy different position spaces."""

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)          # (max_len, d_head/2)
        emb = torch.cat([freqs, freqs], dim=-1)                # (max_len, d_head)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        # q: (B, Hq, L, Dh)  k: (B, Hk, L, Dh) -- same L, same Dh
        L = q.size(2)
        if L > self.cos_cached.size(0):
            raise ValueError(f"sequence length {L} exceeds RoPE max_len {self.cos_cached.size(0)}")
        cos = self.cos_cached[:L].to(q.dtype).to(q.device)[None, None, :, :]
        sin = self.sin_cached[:L].to(q.dtype).to(q.device)[None, None, :, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ==============================================================================
# Normalization: LayerNorm + RMSNorm
# (from src/models/norm.py)
# ==============================================================================

"""Normalization layers: standard LayerNorm and RMSNorm, both from scratch."""
import torch
import torch.nn as nn


class LayerNormCustom(nn.Module):
    """Standard LayerNorm, implemented from scratch (mean/var + learnable affine)."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm: no mean-centering, no bias term."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ==============================================================================
# Position-wise Feed-Forward Network
# (from src/models/ffn.py)
# ==============================================================================

"""Position-wise Feed-Forward Network."""
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ==============================================================================
# Attention: Scaled Dot-Product, Multi-Head (MHA), Grouped-Query (GQA)
# (from src/models/attention.py)
# ==============================================================================

"""Scaled Dot-Product Attention, Multi-Head Attention (MHA), and Grouped-Query
Attention (GQA) -- all from scratch (no nn.MultiheadAttention)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V"""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q: (B, H, Lq, Dh)  k,v: (B, H, Lk, Dh)  mask: (B, 1, Lq, Lk) 1=keep 0=mask
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Use the dtype's own minimum, not a hardcoded -1e9: under AMP the
            # matmul above runs in float16 on GPU, and float16's max magnitude
            # (~65504) overflows on a fixed -1e9 constant. finfo(dtype).min is
            # always safely representable and, since every position in a fully
            # masked row gets the same value, softmax still reduces to a
            # harmless uniform distribution there.
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, mask_value)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, use_rope=False, max_len=4096, rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.d_head, max_len, rope_base)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,L,Dh)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        if self.use_rope:
            q, k = self.rope(q, k)
        out, attn = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out), attn


class GroupedQueryAttention(nn.Module):
    """Multi-head queries, but K/V are shared across groups of query heads
    (n_kv_heads < n_heads). Reduces KV projection size / KV-cache footprint."""

    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.0, use_rope=False,
                 max_len=4096, rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model)
        self.attn = ScaledDotProductAttention(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryEmbedding(self.d_head, max_len, rope_base)

    def _repeat_kv(self, x):
        # (B, Hkv, L, Dh) -> (B, Hkv * n_rep, L, Dh)
        if self.n_rep == 1:
            return x
        B, Hkv, L, Dh = x.shape
        x = x[:, :, None, :, :].expand(B, Hkv, self.n_rep, L, Dh)
        return x.reshape(B, Hkv * self.n_rep, L, Dh)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_kv_heads, self.d_head).transpose(1, 2)
        if self.use_rope:
            q, k = self.rope(q, k)
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        out, attn = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.n_heads * self.d_head)
        return self.out_proj(out), attn


# ==============================================================================
# Pre-LN Encoder/Decoder layers and stacks
# (from src/models/layers.py)
# ==============================================================================

"""Pre-LN Encoder/Decoder layers and stacks, assembled from attention.py, ffn.py
and norm.py. Shared by the global (token/patch level) backbone in transformer.py
and the local (within-patch) stacks in blt.py -- not one of the four files
named in the assignment spec, but required to wire MHA/GQA + FFN + norm into
an actual encoder/decoder (see README.md for the full file-to-requirement map)."""
import torch.nn as nn



class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, norm_cls, self_attn_factory):
        super().__init__()
        self.self_attn = self_attn_factory()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, src_mask)
        x = x + self.dropout(attn_out)
        h2 = self.norm2(x)
        x = x + self.dropout(self.ffn(h2))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, norm_cls, self_attn_factory, cross_attn_factory):
        super().__init__()
        self.self_attn = self_attn_factory()
        self.cross_attn = cross_attn_factory()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        h = self.norm1(x)
        sa_out, _ = self.self_attn(h, h, h, tgt_mask)
        x = x + self.dropout(sa_out)
        h2 = self.norm2(x)
        ca_out, attn_w = self.cross_attn(h2, memory, memory, memory_mask)
        x = x + self.dropout(ca_out)
        h3 = self.norm3(x)
        x = x + self.dropout(self.ffn(h3))
        return x, attn_w


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout, norm_cls, self_attn_factory):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, dropout, norm_cls, self_attn_factory) for _ in range(n_layers)]
        )
        self.final_norm = norm_cls(d_model)

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, d_model, d_ff, dropout, norm_cls, self_attn_factory, cross_attn_factory):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, dropout, norm_cls, self_attn_factory, cross_attn_factory)
             for _ in range(n_layers)]
        )
        self.final_norm = norm_cls(d_model)

    def forward(self, x, memory, tgt_mask, memory_mask):
        attn_w = None
        for layer in self.layers:
            x, attn_w = layer(x, memory, tgt_mask, memory_mask)
        return self.final_norm(x), attn_w


# ==============================================================================
# Byte Latent Transformer (BLT): Local Encoder / Local Decoder
# (from src/models/blt.py)
# ==============================================================================

"""Byte Latent Transformer (BLT) style Local Encoder / Local Decoder patch modules.

Design (fixed-size byte patching, patch size P):

Local Encoder (used for the source, and to build "input patches" for the target):
  raw bytes -> byte embedding -> local (within-patch-only) sinusoidal PE -> a
  small local TransformerEncoder that only self-attends WITHIN each patch
  (windowed, O(L*P) not O(L^2)) -> a learnable pooling query cross-attends over
  each patch's byte representations -> one embedding per patch.

Global backbone: an ordinary TransformerEncoder/TransformerDecoder (layers.py,
shared with the token-based configs) operates over the *patch* sequence
instead of a token sequence.

Local Decoder: expands a single patch's global (causal, leakage-free) context
vector back into that patch's P bytes, autoregressively, via a small local
TransformerDecoder with byte-level causal self-attention + cross-attention to
the one patch memory vector.

Leakage-safety: the global decoder's *input* patch sequence is the target
patch-embedding sequence shifted right by ONE WHOLE PATCH (a learnable BOS
patch embedding is prepended, the last patch dropped) -- so
global_decoder_output[j] depends only on target patches < j, never patch j
itself. Within a patch, the Local Decoder also shifts its byte input right by
one (a BOS byte token), so byte i's prediction depends only on bytes < i of
the same patch plus the (already leakage-free) patch memory. This is verified
with a gradient-based leakage test (see dev/test_blt.py in the project's test
suite: the gradient of a mid-sequence logit w.r.t. embeddings of all bytes at
or after that position is exactly zero).
"""
import torch
import torch.nn as nn


PAD_BYTE = 256
BOS_BYTE = 257
EOS_BYTE = 258
BYTE_VOCAB_SIZE = 259  # 0-255 raw bytes + PAD + BOS + EOS


def bytes_to_patch_aligned_tensor(byte_id_lists, patch_size, device=None):
    """byte_id_lists: list of List[int] (already including BOS/EOS).
    Returns a (B, L) LongTensor, right-padded with PAD_BYTE to both the batch
    max length and the next multiple of patch_size."""
    max_len = max(len(x) for x in byte_id_lists)
    pad_len = (-max_len) % patch_size
    L = max_len + pad_len
    out = torch.full((len(byte_id_lists), L), PAD_BYTE, dtype=torch.long, device=device)
    for i, seq in enumerate(byte_id_lists):
        out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


class LocalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, patch_size, dropout=0.1,
                 vocab_size=BYTE_VOCAB_SIZE, pad_id=PAD_BYTE, norm_cls=LayerNormCustom):
        super().__init__()
        self.patch_size = patch_size
        self.pad_id = pad_id
        self.byte_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=patch_size)
        self.local_transformer = TransformerEncoder(
            n_layers=n_layers, d_model=d_model, d_ff=d_ff, dropout=dropout, norm_cls=norm_cls,
            self_attn_factory=lambda: MultiHeadAttention(d_model, n_heads, dropout=dropout),
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, byte_ids):
        """byte_ids: (B, L) with L a multiple of patch_size.
        Returns patch_emb (B, n_patches, D), patch_mask (B, n_patches) bool (True = real content)."""
        B, L = byte_ids.shape
        P = self.patch_size
        n_patches = L // P
        ids = byte_ids.view(B, n_patches, P)
        x = self.byte_emb(ids).view(B * n_patches, P, -1)
        x = self.dropout(self.pos_enc(x))

        byte_keep = ids.ne(self.pad_id).view(B * n_patches, 1, 1, P).float()
        x = self.local_transformer(x, byte_keep)

        q = self.pool_query.expand(B * n_patches, 1, -1)
        patch_emb, _ = self.pool_attn(q, x, x, byte_keep)
        patch_emb = patch_emb.view(B, n_patches, -1)

        patch_mask = ids.ne(self.pad_id).any(dim=-1)  # (B, n_patches)
        return patch_emb, patch_mask


class LocalDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, patch_size, dropout=0.1,
                 vocab_size=BYTE_VOCAB_SIZE, pad_id=PAD_BYTE, bos_id=BOS_BYTE,
                 norm_cls=LayerNormCustom):
        super().__init__()
        self.patch_size = patch_size
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.byte_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=patch_size)
        self.local_decoder = TransformerDecoder(
            n_layers=n_layers, d_model=d_model, d_ff=d_ff, dropout=dropout, norm_cls=norm_cls,
            self_attn_factory=lambda: MultiHeadAttention(d_model, n_heads, dropout=dropout),
            cross_attn_factory=lambda: MultiHeadAttention(d_model, n_heads, dropout=dropout),
        )
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "_causal", torch.tril(torch.ones(patch_size, patch_size))[None, None, :, :], persistent=False
        )

    def shift_target_bytes_within_patch(self, target_byte_ids):
        """target_byte_ids: (B, n_patches, P) ground-truth bytes for each patch.
        Returns decoder input (B, n_patches, P): [BOS, b_0, ..., b_{P-2}] per patch."""
        B, n_patches, P = target_byte_ids.shape
        bos_col = torch.full((B, n_patches, 1), self.bos_id, dtype=torch.long, device=target_byte_ids.device)
        return torch.cat([bos_col, target_byte_ids[:, :, :-1]], dim=-1)

    def forward(self, decoder_input_byte_ids, patch_memory, patch_memory_mask=None):
        """decoder_input_byte_ids: (B, n_patches, P) -- output of shift_target_bytes_within_patch.
        patch_memory: (B, n_patches, D) one leakage-free context vector per patch.
        Returns logits (B, n_patches, P, vocab_size)."""
        B, n_patches, P = decoder_input_byte_ids.shape
        x = self.byte_emb(decoder_input_byte_ids).view(B * n_patches, P, -1)
        x = self.dropout(self.pos_enc(x))

        causal = self._causal.expand(B * n_patches, 1, P, P)
        mem = patch_memory.reshape(B * n_patches, 1, -1)
        if patch_memory_mask is not None:
            mem_mask = patch_memory_mask.reshape(B * n_patches, 1, 1, 1).float()
        else:
            mem_mask = None

        out, _ = self.local_decoder(x, mem, causal, mem_mask)
        logits = self.output_proj(out)
        return logits.view(B, n_patches, P, -1)


def shift_patches_right(patch_emb, bos_patch_param):
    """patch_emb: (B, n_patches, D) target-input patch embeddings (from LocalEncoder
    on the *target* byte stream). Returns (B, n_patches, D): [BOS_patch, p_0, ..., p_{n-2}],
    the leakage-free input sequence for the global decoder."""
    B, n_patches, D = patch_emb.shape
    bos = bos_patch_param.expand(B, 1, D)
    return torch.cat([bos, patch_emb[:, :-1, :]], dim=1)


def shift_mask_right(mask):
    """mask: (B, n_patches) bool/float, True/1 = patch i has real (non-pad) content.
    Returns the same mask re-indexed to line up with shift_patches_right's output:
    position 0 (BOS) is always valid, position k (k>=1) inherits patch (k-1)'s validity.
    Without this, the padding mask applied to the *shifted* decoder-input sequence is
    off by one -- harmless for the training loss (it only ever mislabels already-ignored
    padding positions) but it wrongly hides the most-recently-completed patch from the
    query position that autoregressive generation actually needs at each step."""
    B, n_patches = mask.shape
    bos_valid = torch.ones(B, 1, dtype=mask.dtype, device=mask.device)
    return torch.cat([bos_valid, mask[:, :-1]], dim=1)


# ==============================================================================
# Full model: CipherTransformer + the five ablation configs (C1-C5)
# (from src/models/transformer.py)
# ==============================================================================

"""The full Encoder-Decoder Transformer (CipherTransformer), assembled from
attention.py / positional.py / norm.py / ffn.py / layers.py / blt.py, plus the
five ablation configs (C1-C5) from Table 1 of the assignment."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn



@dataclass
class ModelConfig:
    name: str
    pe_type: str          # 'sinusoidal' | 'rope'
    attn_type: str        # 'mha' | 'gqa'
    norm_type: str        # 'layernorm' | 'rmsnorm'
    tokenization: str     # 'bpe' | 'blt'

    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 1100

    # bpe path
    vocab_src: int = 4000
    vocab_tgt: int = 4000
    pad_id_src: int = 0
    pad_id_tgt: int = 0
    bos_id_tgt: int = 1
    eos_id_tgt: int = 2

    # blt path
    patch_size: int = 4
    n_local_layers: int = 1


CONFIGS = {
    "C1_base": ModelConfig(
        name="C1_base", pe_type="sinusoidal", attn_type="mha", norm_type="layernorm", tokenization="bpe",
    ),
    "C2_rope": ModelConfig(
        name="C2_rope", pe_type="rope", attn_type="mha", norm_type="layernorm", tokenization="bpe",
    ),
    "C3_gqa": ModelConfig(
        name="C3_gqa", pe_type="sinusoidal", attn_type="gqa", norm_type="layernorm", tokenization="bpe",
    ),
    "C4_rmsnorm": ModelConfig(
        name="C4_rmsnorm", pe_type="sinusoidal", attn_type="mha", norm_type="rmsnorm", tokenization="bpe",
    ),
    "C5_blt": ModelConfig(
        name="C5_blt", pe_type="sinusoidal", attn_type="mha", norm_type="layernorm", tokenization="blt",
        d_model=512, d_ff=2048, n_enc_layers=3, n_dec_layers=3, patch_size=4, n_local_layers=1,
        max_len=2816,  # bytes; longest source lines run up to ~2670 chars
    ),
}


class CipherTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        norm_cls = LayerNormCustom if cfg.norm_type == "layernorm" else RMSNorm
        use_rope = cfg.pe_type == "rope"

        def self_attn_factory():
            if cfg.attn_type == "mha":
                return MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout,
                                           use_rope=use_rope, max_len=cfg.max_len)
            return GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, dropout=cfg.dropout,
                                          use_rope=use_rope, max_len=cfg.max_len)

        def cross_attn_factory():
            if cfg.attn_type == "mha":
                return MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, use_rope=False)
            return GroupedQueryAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, dropout=cfg.dropout,
                                          use_rope=False)

        self.encoder = TransformerEncoder(cfg.n_enc_layers, cfg.d_model, cfg.d_ff, cfg.dropout,
                                           norm_cls, self_attn_factory)
        self.decoder = TransformerDecoder(cfg.n_dec_layers, cfg.d_model, cfg.d_ff, cfg.dropout,
                                           norm_cls, self_attn_factory, cross_attn_factory)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.emb_scale = math.sqrt(cfg.d_model)

        if cfg.tokenization == "bpe":
            self.src_tok_emb = nn.Embedding(cfg.vocab_src, cfg.d_model, padding_idx=cfg.pad_id_src)
            self.tgt_tok_emb = nn.Embedding(cfg.vocab_tgt, cfg.d_model, padding_idx=cfg.pad_id_tgt)
            self.out_proj = nn.Linear(cfg.d_model, cfg.vocab_tgt)
            self.out_proj.weight = self.tgt_tok_emb.weight  # weight tying
            if not use_rope:
                self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        else:
            self.src_local_enc = LocalEncoder(cfg.d_model, cfg.n_heads, cfg.n_local_layers, cfg.d_ff,
                                               cfg.patch_size, cfg.dropout, norm_cls=norm_cls)
            self.tgt_local_enc = LocalEncoder(cfg.d_model, cfg.n_heads, cfg.n_local_layers, cfg.d_ff,
                                               cfg.patch_size, cfg.dropout, norm_cls=norm_cls)
            self.local_dec = LocalDecoder(cfg.d_model, cfg.n_heads, cfg.n_local_layers, cfg.d_ff,
                                           cfg.patch_size, cfg.dropout, norm_cls=norm_cls)
            self.bos_patch = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
            n_patches_max = cfg.max_len // cfg.patch_size + 2
            self.patch_pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=n_patches_max)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0.0)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- BPE path ----------------

    def forward_bpe(self, src_ids, tgt_in_ids, src_mask, tgt_mask):
        src = self.src_tok_emb(src_ids) * self.emb_scale
        tgt = self.tgt_tok_emb(tgt_in_ids) * self.emb_scale
        if self.cfg.pe_type == "sinusoidal":
            src = self.pos_enc(src)
            tgt = self.pos_enc(tgt)
        src = self.emb_dropout(src)
        tgt = self.emb_dropout(tgt)
        memory = self.encoder(src, src_mask)
        dec_out, attn_w = self.decoder(tgt, memory, tgt_mask, src_mask)
        logits = self.out_proj(dec_out)
        return logits, attn_w

    def encode_bpe(self, src_ids, src_mask):
        src = self.src_tok_emb(src_ids) * self.emb_scale
        if self.cfg.pe_type == "sinusoidal":
            src = self.pos_enc(src)
        src = self.emb_dropout(src)
        return self.encoder(src, src_mask)

    def decode_step_bpe(self, tgt_in_ids, memory, tgt_mask, src_mask):
        tgt = self.tgt_tok_emb(tgt_in_ids) * self.emb_scale
        if self.cfg.pe_type == "sinusoidal":
            tgt = self.pos_enc(tgt)
        tgt = self.emb_dropout(tgt)
        dec_out, attn_w = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.out_proj(dec_out), attn_w

    # ---------------- BLT path ----------------

    def encode_blt(self, src_byte_ids):
        src_patch_emb, src_patch_mask = self.src_local_enc(src_byte_ids)
        src_patch_emb = self.patch_pos_enc(src_patch_emb)
        src_attn_mask = src_patch_mask[:, None, None, :].float()
        memory = self.encoder(src_patch_emb, src_attn_mask)
        return memory, src_attn_mask

    def forward_blt(self, src_byte_ids, tgt_byte_ids):
        """Teacher-forced training forward. tgt_byte_ids: (B, L) patch-aligned,
        already including leading BOS_BYTE / trailing EOS_BYTE / PAD_BYTE padding.
        Returns logits (B, L, BYTE_VOCAB_SIZE)."""
        B, L = tgt_byte_ids.shape
        P = self.cfg.patch_size
        n_patches = L // P

        memory, src_attn_mask = self.encode_blt(src_byte_ids)

        tgt_input_patch_emb, tgt_patch_mask = self.tgt_local_enc(tgt_byte_ids)
        tgt_input_patch_emb = shift_patches_right(tgt_input_patch_emb, self.bos_patch)
        tgt_input_patch_emb = self.patch_pos_enc(tgt_input_patch_emb)

        causal = torch.tril(torch.ones(n_patches, n_patches, device=tgt_byte_ids.device))[None, None, :, :]
        pad = shift_mask_right(tgt_patch_mask)[:, None, None, :].float()
        tgt_causal_mask = causal * pad

        patch_memory, _ = self.decoder(tgt_input_patch_emb, memory, tgt_causal_mask, src_attn_mask)

        tgt_ids_grouped = tgt_byte_ids.view(B, n_patches, P)
        local_dec_input = self.local_dec.shift_target_bytes_within_patch(tgt_ids_grouped)
        logits = self.local_dec(local_dec_input, patch_memory, tgt_patch_mask)
        return logits.view(B, L, -1)


def build_model(config_name: str) -> CipherTransformer:
    return CipherTransformer(CONFIGS[config_name])


# ==============================================================================
# Data loading, bit<->byte<->char packing, BPE tokenizer training, Datasets
# (from src/dataset.py)
# ==============================================================================

"""Data loading, cipher <-> byte <-> BPE-friendly-char packing, train/val/test
splitting, BPE tokenizer training, and the two parallel Dataset/collate
pipelines: tokenized (BPE, for C1-C4) and token-free (BLT, for C5)."""
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import TemplateProcessing


SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

BYTE_CHAR_OFFSET = 0x2400  # Unicode "control pictures" block, 256 consecutive printable-safe chars


# ---------------------------------------------------------------------------
# bit <-> byte <-> char packing
# ---------------------------------------------------------------------------

def bits_to_byte_ints(bitstr: str):
    assert len(bitstr) % 8 == 0, f"cipher length {len(bitstr)} not a multiple of 8"
    return [int(bitstr[i:i + 8], 2) for i in range(0, len(bitstr), 8)]


def byte_ints_to_bits(byte_ints):
    return "".join(f"{b:08b}" for b in byte_ints)


def byte_ints_to_char_str(byte_ints):
    return "".join(chr(BYTE_CHAR_OFFSET + b) for b in byte_ints)


def char_str_to_byte_ints(s):
    return [ord(ch) - BYTE_CHAR_OFFSET for ch in s]


def load_aligned_lines(plain_path, cipher_path):
    plain = Path(plain_path).read_text(encoding="utf-8").splitlines()
    cipher = Path(cipher_path).read_text(encoding="utf-8").splitlines()
    assert len(plain) == len(cipher), (len(plain), len(cipher))
    for p, c in zip(plain, cipher):
        assert len(c) == 8 * len(p), "8-bits-per-character invariant violated"
        assert set(c) <= {"0", "1"}
    return plain, cipher


def make_split(n, seed=42, ratios=(0.8, 0.1, 0.1)):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": sorted(idx[:n_train]),
        "val": sorted(idx[n_train:n_train + n_val]),
        "test": sorted(idx[n_train + n_val:]),
    }


# ---------------------------------------------------------------------------
# BPE tokenizer training (C1-C4 only; C5/BLT operates on raw bytes)
# ---------------------------------------------------------------------------

def train_bpe_tokenizer(texts, vocab_size, use_whitespace_pretok):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    if use_whitespace_pretok:
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        # cipher byte-char stream: no word boundaries, let BPE merge freely.
        # decode() must FUSE tokens with no separator (not the tokenizers-library
        # default of joining with " ") or the reconstructed byte stream is corrupted.
        tok.pre_tokenizer = None
        tok.decoder = decoders.Fuse()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIALS, min_frequency=2)
    tok.train_from_iterator(texts, trainer=trainer)
    tok.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tok.token_to_id("[BOS]")), ("[EOS]", tok.token_to_id("[EOS]"))],
    )
    return tok


# ---------------------------------------------------------------------------
# Tokenized (BPE) pipeline -- C1, C2, C3, C4
# ---------------------------------------------------------------------------

class CipherBPEDataset(Dataset):
    """Pre-tokenizes plain text (source) and byte-packed cipher (target) with
    trained BPE tokenizers. Each item already includes BOS/EOS (added by the
    tokenizer's post-processor)."""

    def __init__(self, plain_lines, cipher_lines, indices, src_tok, tgt_tok, max_len):
        self.src_ids, self.tgt_ids = [], []
        for i in indices:
            s = src_tok.encode(plain_lines[i]).ids[:max_len]
            t_char = byte_ints_to_char_str(bits_to_byte_ints(cipher_lines[i]))
            t = tgt_tok.encode(t_char).ids[:max_len]
            self.src_ids.append(s)
            self.tgt_ids.append(t)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return self.src_ids[idx], self.tgt_ids[idx]


def collate_bpe(batch, pad_id=PAD_ID):
    src_seqs, tgt_seqs = zip(*batch)
    Ls = max(len(s) for s in src_seqs)
    Lt_full = max(len(t) for t in tgt_seqs)

    B = len(batch)
    src = torch.full((B, Ls), pad_id, dtype=torch.long)
    tgt_full = torch.full((B, Lt_full), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_full[i, :len(t)] = torch.tensor(t, dtype=torch.long)

    tgt_in = tgt_full[:, :-1]
    tgt_out = tgt_full[:, 1:]
    Lt = tgt_in.size(1)

    src_lens = torch.tensor([len(s) for s in src_seqs])
    tgt_lens = torch.tensor([len(t) - 1 for t in tgt_seqs])  # length of decoder input part

    src_mask = (torch.arange(Ls)[None, :] < src_lens[:, None]).float()[:, None, None, :]
    tgt_pad_mask = (torch.arange(Lt)[None, :] < tgt_lens[:, None]).float()[:, None, None, :]
    causal = torch.tril(torch.ones(Lt, Lt))[None, None, :, :]
    tgt_mask = causal * tgt_pad_mask

    return {
        "src_ids": src, "tgt_in_ids": tgt_in, "tgt_out_ids": tgt_out,
        "src_mask": src_mask, "tgt_mask": tgt_mask,
    }


# ---------------------------------------------------------------------------
# Token-free (BLT) pipeline -- C5
# ---------------------------------------------------------------------------

class CipherBLTDataset(Dataset):
    """Raw byte-id sequences (with BOS/EOS), no subword tokenization."""

    def __init__(self, plain_lines, cipher_lines, indices, max_len_bytes):
        self.src_bytes, self.tgt_bytes = [], []
        for i in indices:
            s = [BOS_BYTE] + list(plain_lines[i].encode("utf-8")) + [EOS_BYTE]
            byte_ints = bits_to_byte_ints(cipher_lines[i])
            t = [BOS_BYTE] + byte_ints + [EOS_BYTE]
            self.src_bytes.append(s[:max_len_bytes])
            self.tgt_bytes.append(t[:max_len_bytes])

    def __len__(self):
        return len(self.src_bytes)

    def __getitem__(self, idx):
        return self.src_bytes[idx], self.tgt_bytes[idx]


def collate_blt(batch, patch_size):
    src_seqs, tgt_seqs = zip(*batch)
    src = bytes_to_patch_aligned_tensor(list(src_seqs), patch_size)
    tgt = bytes_to_patch_aligned_tensor(list(tgt_seqs), patch_size)
    return {"src_byte_ids": src, "tgt_byte_ids": tgt}


# ==============================================================================
# Greedy-decoding generation, evaluation metrics, results plotting
# (from src/utils.py)
# ==============================================================================

"""Greedy-decoding generation, the full evaluation metric suite (Bit-Level
Accuracy, Sequence Accuracy, Levenshtein Distance, BLEU/ROUGE), and results
plotting."""
from pathlib import Path

import torch
from rapidfuzz.distance import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


_BLEU_SMOOTHING = SmoothingFunction().method1
_ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)


# ---------------------------------------------------------------------------
# Greedy (argmax) autoregressive generation -- required so results are
# consistent/comparable across all five configs. No KV-caching (recomputes the
# growing prefix each step): simple and easy to verify correct, but O(n^2) per
# sequence, so generation-based eval is run on a bounded subset of the test set.
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate_bpe(model, src_ids, src_mask, max_len, device,
                         bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID):
    model.eval()
    B = src_ids.size(0)
    memory = model.encode_bpe(src_ids, src_mask)
    generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        Lt = generated.size(1)
        causal = torch.tril(torch.ones(Lt, Lt, device=device))[None, None, :, :]
        logits, _ = model.decode_step_bpe(generated, memory, causal, src_mask)
        next_tok = logits[:, -1, :].argmax(-1)
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        generated = torch.cat([generated, next_tok.unsqueeze(1)], dim=1)
        finished = finished | (next_tok == eos_id)
        if finished.all():
            break
    return generated  # (B, <=max_len+1) includes leading BOS


@torch.no_grad()
def greedy_generate_blt(model, src_byte_ids, max_patches, device):
    """Patch-by-patch autoregressive generation."""
    model.eval()
    B = src_byte_ids.size(0)
    P = model.cfg.patch_size
    memory, src_attn_mask = model.encode_blt(src_byte_ids)

    all_bytes = torch.full((B, max_patches * P), PAD_BYTE, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    j = 0
    for j in range(max_patches):
        cur_len = (j + 1) * P
        cur_bytes = all_bytes[:, :cur_len]

        tgt_input_patch_emb, tgt_patch_mask = model.tgt_local_enc(cur_bytes)
        tgt_input_patch_emb = shift_patches_right(tgt_input_patch_emb, model.bos_patch)
        tgt_input_patch_emb = model.patch_pos_enc(tgt_input_patch_emb)

        n_p = j + 1
        causal = torch.tril(torch.ones(n_p, n_p, device=device))[None, None, :, :]
        pad = shift_mask_right(tgt_patch_mask)[:, None, None, :].float()
        tgt_causal_mask = causal * pad

        patch_memory_all, _ = model.decoder(tgt_input_patch_emb, memory, tgt_causal_mask, src_attn_mask)
        patch_memory_j = patch_memory_all[:, j:j + 1, :]  # context for generating patch j, leakage-free

        patch_bytes = torch.full((B, P), PAD_BYTE, dtype=torch.long, device=device)
        for t in range(P):
            dec_in = model.local_dec.shift_target_bytes_within_patch(patch_bytes.unsqueeze(1)).squeeze(1)
            logits = model.local_dec(dec_in.unsqueeze(1), patch_memory_j)
            next_byte = logits[:, 0, t, :].argmax(-1)
            patch_bytes[:, t] = torch.where(finished, torch.full_like(next_byte, PAD_BYTE), next_byte)

        all_bytes[:, j * P:(j + 1) * P] = patch_bytes
        finished = finished | (patch_bytes == EOS_BYTE).any(dim=-1)
        if finished.all():
            break

    return all_bytes[:, :(j + 1) * P]


def strip_bos_eos(token_ids, bos_id, eos_id):
    """token_ids: 1D list/tensor. Drops a leading BOS and truncates at the first
    EOS (exclusive)."""
    ids = [i for i in (token_ids.tolist() if torch.is_tensor(token_ids) else token_ids)]
    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if eos_id in ids:
        ids = ids[:ids.index(eos_id)]
    return ids


def decode_bpe_ids_to_bits(token_ids, tgt_tokenizer, eos_id=EOS_ID, bos_id=BOS_ID):
    """token_ids: 1D list/tensor of generated target-vocab ids (leading BOS included).
    Returns the reconstructed cipher bit-string (best-effort; malformed/incomplete
    generations may not decode to a clean multiple of 8 bits)."""
    ids = strip_bos_eos(token_ids, bos_id, eos_id)
    char_str = tgt_tokenizer.decode(ids, skip_special_tokens=True)
    try:
        byte_ints = char_str_to_byte_ints(char_str)
    except Exception:
        return ""
    return byte_ints_to_bits(byte_ints)


def gold_bpe_token_ids(gold_bits, tgt_tokenizer, eos_id=EOS_ID, bos_id=BOS_ID):
    """The reference token sequence for BLEU/ROUGE: how the tokenizer itself
    would encode the true cipher, with BOS/EOS stripped to match the
    hypothesis side (strip_bos_eos on generated output)."""
    char_str = byte_ints_to_char_str(bits_to_byte_ints(gold_bits))
    ids = tgt_tokenizer.encode(char_str).ids
    return strip_bos_eos(ids, bos_id, eos_id)


def decode_blt_bytes_to_bits(byte_ids):
    ids = byte_ids.tolist() if torch.is_tensor(byte_ids) else list(byte_ids)
    if ids and ids[0] == BOS_BYTE:
        ids = ids[1:]
    if EOS_BYTE in ids:
        ids = ids[:ids.index(EOS_BYTE)]
    ids = [b for b in ids if b != PAD_BYTE]
    return byte_ints_to_bits(ids)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def bit_level_metrics(predicted_bits, gold_bits):
    """Returns (bit_accuracy, exact_match) for one example, comparing on the
    overlapping prefix and penalizing any length mismatch as extra errors."""
    L = max(len(predicted_bits), len(gold_bits))
    if L == 0:
        return 1.0, True
    matches = sum(1 for a, b in zip(predicted_bits, gold_bits) if a == b)
    bit_acc = matches / L
    exact = predicted_bits == gold_bits
    return bit_acc, exact


def bleu_score(hyp_tokens, ref_tokens):
    """Sentence-level BLEU (n-gram precision, up to 4-grams, with smoothing).
    hyp_tokens/ref_tokens are lists of BPE token ids treated as opaque "words"
    -- meaningful for tokenized (BPE) models only."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    hyp_str = [str(t) for t in hyp_tokens]
    ref_str = [str(t) for t in ref_tokens]
    return sentence_bleu([ref_str], hyp_str, smoothing_function=_BLEU_SMOOTHING)


def rouge_scores(hyp_tokens, ref_tokens):
    """ROUGE-1/2/L F-measures, same token-id-as-word convention as bleu_score
    (BPE-tokenized models only)."""
    hyp_str = " ".join(str(t) for t in hyp_tokens)
    ref_str = " ".join(str(t) for t in ref_tokens)
    scores = _ROUGE_SCORER.score(ref_str, hyp_str)
    return {k: v.fmeasure for k, v in scores.items()}


def compute_all_metrics(predicted_bits, gold_bits, hyp_token_ids=None, ref_token_ids=None):
    """One evaluation example's full metric suite, all from greedy-decoded output:
      - bit_acc            Bit-Level Accuracy.
      - sequence_acc        Sequence Accuracy (1.0 iff bit-for-bit exact).
      - levenshtein         Raw edit distance between predicted/gold bit strings.
      - levenshtein_norm    Edit distance normalized by max(len(pred), len(gold)).
      - bleu, rouge1/2/L    Only when hyp_token_ids/ref_token_ids are given
                            (tokenized C1-C4 models); None for C5 (BLT), per spec.
    """
    bit_acc, exact = bit_level_metrics(predicted_bits, gold_bits)
    lev = Levenshtein.distance(predicted_bits, gold_bits)
    lev_norm = lev / max(len(predicted_bits), len(gold_bits), 1)

    out = {
        "bit_acc": bit_acc,
        "sequence_acc": float(exact),
        "levenshtein": lev,
        "levenshtein_norm": lev_norm,
        "bleu": None, "rouge1": None, "rouge2": None, "rougeL": None,
    }
    if hyp_token_ids is not None and ref_token_ids is not None:
        out["bleu"] = bleu_score(hyp_token_ids, ref_token_ids)
        out.update(rouge_scores(hyp_token_ids, ref_token_ids))
    return out


# ---------------------------------------------------------------------------
# Results plotting
# ---------------------------------------------------------------------------

def plot_ablation_results(results, out_path, config_order=None):
    """results: dict config_name -> flat metrics dict (as produced by
    train.run_config / train.evaluate_config). Saves a 3x3 grid PNG comparing
    the five configs across the required metrics."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results).T
    df.index.name = "config"
    if config_order is not None:
        df = df.reindex(config_order)

    metrics_to_plot = [
        ("test_loss", "Test loss"), ("test_ppl", "Test perplexity"), ("n_params", "Parameters"),
        ("gen_bit_acc", "Bit-Level Accuracy"), ("gen_sequence_acc", "Sequence Accuracy"),
        ("gen_levenshtein_norm", "Levenshtein distance (normalized)"),
        ("gen_bleu", "BLEU (tokenized only)"), ("gen_rouge1", "ROUGE-1 (tokenized only)"),
        ("gen_rougeL", "ROUGE-L (tokenized only)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (col, title) in zip(axes.flat, metrics_to_plot):
        if col not in df.columns:
            ax.set_title(title)
            continue
        vals = df[col].astype(float)
        ax.bar(df.index, vals.fillna(0))
        for idx, v in enumerate(vals):
            if pd.isna(v):
                ax.text(idx, 0, "n/a", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return df


# ==============================================================================
# Training loop, W&B logging, HF Hub push, CLI entry point
# (from src/train.py)
# ==============================================================================

"""Main training loop: builds one (or all five) ablation config(s), trains with
Weights & Biases logging, evaluates (teacher-forced + bounded greedy-decoding
generation metrics), saves a checkpoint, optionally pushes it to the Hugging
Face Hub, and records W&B run URLs / HF repo URLs ("soft links") to
outputs/LINKS.md.

Usage:
    python src/train.py --config C1_base
    python src/train.py --config all --epochs 40 --push-to-hub
"""
import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


ALL_CONFIGS = ["C1_base", "C2_rope", "C3_gqa", "C4_rmsnorm", "C5_blt"]


# ---------------------------------------------------------------------------
# Optimizer / schedule / train-eval loops
# ---------------------------------------------------------------------------

def noam_lr_lambda(step, d_model, warmup_steps):
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def build_optimizer_and_scheduler(model, d_model, warmup_steps=4000, betas=(0.9, 0.98), eps=1e-9):
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_lr_lambda(step, d_model, warmup_steps)
    )
    return optimizer, scheduler


def _forward_and_targets(model, batch, pipeline):
    if pipeline == "bpe":
        logits, _ = model.forward_bpe(batch["src_ids"], batch["tgt_in_ids"], batch["src_mask"], batch["tgt_mask"])
        targets = batch["tgt_out_ids"]
    else:
        logits = model.forward_blt(batch["src_byte_ids"], batch["tgt_byte_ids"])
        targets = batch["tgt_byte_ids"]
    return logits, targets


def train_one_epoch(model, loader, optimizer, scheduler, device, pipeline, pad_id,
                     grad_clip=1.0, use_amp=False, label_smoothing=0.1, log_fn=None, global_step=0):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    total_loss, total_tokens = 0.0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, targets = _forward_and_targets(model, batch, pipeline)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        n_tok = (targets != pad_id).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

        if log_fn is not None:
            log_fn({
                "train/loss_step": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": float(grad_norm),
            }, step=global_step)

    return total_loss / max(total_tokens, 1), global_step


@torch.no_grad()
def evaluate(model, loader, device, pipeline, pad_id):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.0)
    total_loss, total_tokens, total_correct = 0.0, 0, 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, targets = _forward_and_targets(model, batch, pipeline)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        mask = targets != pad_id
        preds = logits.argmax(-1)
        total_correct += ((preds == targets) & mask).sum().item()
        n_tok = mask.sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok

    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, 20)),
        "token_acc": total_correct / max(total_tokens, 1),
    }


def save_checkpoint(model, cfg, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "config": cfg.__dict__}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def push_checkpoint_to_hub(local_ckpt_path, repo_id, token, commit_message="add checkpoint", extra_files=None):
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model")
    api.upload_file(
        path_or_fileobj=str(local_ckpt_path), path_in_repo=Path(local_ckpt_path).name,
        repo_id=repo_id, repo_type="model", commit_message=commit_message,
    )
    for f in (extra_files or []):
        api.upload_file(
            path_or_fileobj=str(f), path_in_repo=Path(f).name,
            repo_id=repo_id, repo_type="model", commit_message=commit_message,
        )
    return f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# Dataloaders / greedy-decoding generation eval
# ---------------------------------------------------------------------------

def make_dataloaders_for_config(cfg, plain, cipher, split, src_tok, tgt_tok, batch_size_bpe, batch_size_blt):
    if cfg.tokenization == "bpe":
        train_ds = CipherBPEDataset(plain, cipher, split["train"], src_tok, tgt_tok, max_len=cfg.max_len)
        val_ds = CipherBPEDataset(plain, cipher, split["val"], src_tok, tgt_tok, max_len=cfg.max_len)
        test_ds = CipherBPEDataset(plain, cipher, split["test"], src_tok, tgt_tok, max_len=cfg.max_len)
        collate = collate_bpe
        bs = batch_size_bpe
    else:
        train_ds = CipherBLTDataset(plain, cipher, split["train"], max_len_bytes=cfg.max_len)
        val_ds = CipherBLTDataset(plain, cipher, split["val"], max_len_bytes=cfg.max_len)
        test_ds = CipherBLTDataset(plain, cipher, split["test"], max_len_bytes=cfg.max_len)
        collate = lambda b: collate_blt(b, cfg.patch_size)
        bs = batch_size_blt

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, collate_fn=collate)
    return train_dl, val_dl, test_dl


@torch.no_grad()
def run_generation_eval(model, cfg, plain, cipher, indices, src_tok, tgt_tok, pipeline, device, gen_max_extra):
    # All numbers here come from GREEDY decoding, so results are directly
    # comparable across configs. BLEU/ROUGE only computed for tokenized (BPE)
    # configs; stay None for C5 (BLT, token-free), per spec.
    model.eval()
    per_example = []
    for i in indices:
        gold_bits = cipher[i]
        hyp_token_ids = ref_token_ids = None
        if pipeline == "bpe":
            ids = src_tok.encode(plain[i]).ids[:cfg.max_len]
            src_ids = torch.tensor([ids], dtype=torch.long, device=device)
            src_mask = torch.ones(1, 1, 1, len(ids), device=device)
            ref_token_ids = gold_bpe_token_ids(gold_bits, tgt_tok)
            max_len = min(cfg.max_len, len(ref_token_ids) + gen_max_extra)
            generated = greedy_generate_bpe(model, src_ids, src_mask, max_len=max_len, device=device)
            hyp_token_ids = strip_bos_eos(generated[0], BOS_ID, EOS_ID)
            pred_bits = decode_bpe_ids_to_bits(generated[0], tgt_tok)
        else:
            src_bytes = ([BOS_BYTE] + list(plain[i].encode("utf-8")) + [EOS_BYTE])[:cfg.max_len]
            src_t = bytes_to_patch_aligned_tensor([src_bytes], cfg.patch_size, device=device)
            true_n_patches = math.ceil((len(gold_bits) // 8 + 2) / cfg.patch_size)
            max_patches = true_n_patches + gen_max_extra
            generated = greedy_generate_blt(model, src_t, max_patches=max_patches, device=device)
            pred_bits = decode_blt_bytes_to_bits(generated[0])
        per_example.append(compute_all_metrics(pred_bits, gold_bits, hyp_token_ids, ref_token_ids))

    def avg(key):
        vals = [e[key] for e in per_example if e[key] is not None]
        return statistics.mean(vals) if vals else None

    return {
        "bit_acc": avg("bit_acc") or 0.0,
        "sequence_acc": avg("sequence_acc") or 0.0,
        "levenshtein": avg("levenshtein") or 0.0,
        "levenshtein_norm": avg("levenshtein_norm") or 0.0,
        "bleu": avg("bleu"),
        "rouge1": avg("rouge1"),
        "rouge2": avg("rouge2"),
        "rougeL": avg("rougeL"),
        "n": len(indices),
    }


def _fmt(x, spec=".4f"):
    return "-" if x is None else format(x, spec)


def write_links_manifest(links, results, configs, out_dir):
    lines = [
        "# Soft Links: W&B Runs and Hugging Face Checkpoints", "",
        "All generation-based metrics below use **greedy decoding**. BLEU/ROUGE "
        "are computed over BPE token ids and are only defined for the tokenized "
        "configs (C1-C4); C5 (BLT) is token-free, so those columns are `-`.", "",
        "| Config | W&B Run | HF Checkpoint | Params | Test Loss | Test PPL | "
        "Bit Acc | Seq Acc | Levenshtein | Levenshtein (norm) | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in configs:
        if name not in links:
            continue
        l, r = links[name], results.get(name, {})
        lines.append(
            f"| {name} | {l.get('wandb_url') or '-'} | {l.get('hf_url') or '-'} | "
            f"{r.get('n_params', float('nan')):,.0f} | {r.get('test_loss', float('nan')):.4f} | "
            f"{r.get('test_ppl', float('nan')):.2f} | {_fmt(r.get('gen_bit_acc'))} | "
            f"{_fmt(r.get('gen_sequence_acc'))} | {_fmt(r.get('gen_levenshtein'), '.2f')} | "
            f"{_fmt(r.get('gen_levenshtein_norm'))} | {_fmt(r.get('gen_bleu'))} | "
            f"{_fmt(r.get('gen_rouge1'))} | {_fmt(r.get('gen_rouge2'))} | {_fmt(r.get('gen_rougeL'))} |"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "LINKS.md").write_text("\n".join(lines) + "\n")
    (out_dir / "LINKS.json").write_text(json.dumps({"links": links, "results": results}, indent=2))


# ---------------------------------------------------------------------------
# run_config: train + evaluate + (optionally) push one config end-to-end
# ---------------------------------------------------------------------------

def run_config(config_name, plain, cipher, split, src_tok, tgt_tok, args, device,
               results, links, hf_username=None):
    print(f"\n{'=' * 70}\nConfig: {config_name}\n{'=' * 70}")
    cfg = CONFIGS[config_name]
    if cfg.tokenization == "bpe":
        cfg.vocab_src = src_tok.get_vocab_size()
        cfg.vocab_tgt = tgt_tok.get_vocab_size()

    model = CipherTransformer(cfg).to(device)
    n_params = model.num_parameters()
    print(f"parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    train_dl, val_dl, test_dl = make_dataloaders_for_config(
        cfg, plain, cipher, split, src_tok, tgt_tok, args.batch_size_bpe, args.batch_size_blt
    )
    pipeline = cfg.tokenization
    pad_id = PAD_ID if pipeline == "bpe" else PAD_BYTE

    # Scale warmup to this run's actual step budget (not a fixed absolute count):
    # a fixed 2000-step warmup convention from 100k+-step NMT runs would eat over
    # half of a short few-thousand-step run, so the LR would never reach its peak.
    total_steps = len(train_dl) * args.epochs
    warmup_steps = max(50, min(args.warmup_steps_cap, int(0.06 * total_steps)))
    print(f"total_steps={total_steps}  warmup_steps={warmup_steps} ({100 * warmup_steps / total_steps:.1f}%)")
    opt, sched = build_optimizer_and_scheduler(model, cfg.d_model, warmup_steps=warmup_steps)

    wandb_url = None
    run = None
    if args.wandb_mode != "disabled":
        import wandb
        try:
            run = wandb.init(project=args.wandb_project, name=config_name, mode=args.wandb_mode,
                              config={**cfg.__dict__, "n_params": n_params}, reinit="finish_previous")
            wandb_url = run.url
            print("W&B run:", wandb_url)
        except Exception as e:
            # A W&B outage/auth/network hiccup must not sink hours of training --
            # fall back to running without W&B logging rather than crashing.
            # Re-run this config later with --wandb-mode offline (sync afterward
            # with `wandb sync`) if this keeps happening.
            print(f"[warning] wandb.init failed ({e!r}); continuing WITHOUT W&B logging for {config_name}")
            run = None

    def log_fn(d, step):
        if run is not None:
            run.log(d, step=step)

    global_step = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_dl, opt, sched, device, pipeline, pad_id,
            grad_clip=args.grad_clip, use_amp=(device.type == "cuda"),
            log_fn=log_fn, global_step=global_step,
        )
        val_metrics = evaluate(model, val_dl, device, pipeline, pad_id)
        if run is not None:
            run.log({
                "epoch": epoch, "train/loss_epoch": train_loss, "val/loss": val_metrics["loss"],
                "val/ppl": val_metrics["ppl"], "val/token_acc": val_metrics["token_acc"],
            }, step=global_step)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss {train_loss:.4f}  val_loss {val_metrics['loss']:.4f}  "
              f"val_ppl {val_metrics['ppl']:.2f}  val_token_acc {val_metrics['token_acc']:.4f}")
    train_time_sec = time.time() - t0

    test_metrics = evaluate(model, test_dl, device, pipeline, pad_id)
    print(f"TEST  loss {test_metrics['loss']:.4f}  ppl {test_metrics['ppl']:.2f}  "
          f"token_acc {test_metrics['token_acc']:.4f}")

    gen_metrics = run_generation_eval(
        model, cfg, plain, cipher, split["test"][:args.n_gen_eval_samples],
        src_tok, tgt_tok, pipeline, device, args.gen_max_extra,
    )
    print(f"GEN (greedy decoding, n={gen_metrics['n']})  bit_acc {gen_metrics['bit_acc']:.4f}  "
          f"sequence_acc {gen_metrics['sequence_acc']:.4f}  levenshtein {gen_metrics['levenshtein']:.2f}  "
          f"levenshtein_norm {gen_metrics['levenshtein_norm']:.4f}  bleu {_fmt(gen_metrics['bleu'])}  "
          f"rouge1 {_fmt(gen_metrics['rouge1'])}  rouge2 {_fmt(gen_metrics['rouge2'])}  "
          f"rougeL {_fmt(gen_metrics['rougeL'])}")

    if run is not None:
        wandb_log_payload = {
            "test/loss": test_metrics["loss"], "test/ppl": test_metrics["ppl"],
            "test/token_acc": test_metrics["token_acc"], "test/bit_acc": gen_metrics["bit_acc"],
            "test/sequence_acc": gen_metrics["sequence_acc"], "test/levenshtein": gen_metrics["levenshtein"],
            "test/levenshtein_norm": gen_metrics["levenshtein_norm"], "train_time_sec": train_time_sec,
        }
        if gen_metrics["bleu"] is not None:
            wandb_log_payload.update({
                "test/bleu": gen_metrics["bleu"], "test/rouge1": gen_metrics["rouge1"],
                "test/rouge2": gen_metrics["rouge2"], "test/rougeL": gen_metrics["rougeL"],
            })
        run.log(wandb_log_payload)

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_path = save_checkpoint(
        model, cfg, ckpt_dir / f"{config_name}.pt",
        extra={"test_metrics": test_metrics, "gen_metrics": gen_metrics, "n_params": n_params},
    )

    hf_url = None
    if args.push_to_hub and hf_username is not None:
        repo_id = f"{hf_username}/{args.hf_repo_prefix}-{config_name.replace('_', '-')}"
        hf_url = push_checkpoint_to_hub(
            ckpt_path, repo_id, token=os.environ["HF_TOKEN"],
            commit_message=f"{config_name}: test_loss={test_metrics['loss']:.4f}",
        )
        print("HF checkpoint:", hf_url)

    if run is not None:
        run.finish()

    results[config_name] = {
        "n_params": n_params, "train_time_sec": train_time_sec,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"gen_{k}": v for k, v in gen_metrics.items()},
    }
    links[config_name] = {"wandb_url": wandb_url, "hf_url": hf_url, "checkpoint_local": str(ckpt_path)}
    write_links_manifest(links, results, ALL_CONFIGS, args.output_dir)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="C1_base", choices=ALL_CONFIGS + ["all"],
                    help="which ablation config to train, or 'all' for all five")
    p.add_argument("--plain-path", default="brown_plain.txt")
    p.add_argument("--cipher-path", default="brown_cipher.txt")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size-bpe", type=int, default=16)
    p.add_argument("--batch-size-blt", type=int, default=8)
    p.add_argument("--warmup-steps-cap", type=int, default=2000,
                    help="cap on warmup steps; the real warmup is scaled to ~6%% of "
                         "this run's total steps (len(train_dl)*epochs), capped here")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--vocab-src", type=int, default=4000)
    p.add_argument("--vocab-tgt", type=int, default=4000)
    p.add_argument("--n-gen-eval-samples", type=int, default=100,
                    help="bounded free-running greedy-decoding generation eval set size")
    p.add_argument("--gen-max-extra", type=int, default=24,
                    help="safety margin (tokens/patches) added on top of the known gold length")
    p.add_argument("--wandb-project", default="anlp-a1-cipher-transformer")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hf-repo-prefix", default="anlp-a1-cipher")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    plain, cipher = load_aligned_lines(args.plain_path, args.cipher_path)
    print(f"loaded {len(plain)} aligned lines")
    split = make_split(len(plain), seed=args.seed, ratios=(0.8, 0.1, 0.1))
    print({k: len(v) for k, v in split.items()})

    configs_to_run = ALL_CONFIGS if args.config == "all" else [args.config]
    needs_bpe = any(CONFIGS[c].tokenization == "bpe" for c in configs_to_run)

    src_tok = tgt_tok = None
    if needs_bpe:
        train_plain = [plain[i] for i in split["train"]]
        train_cipher_chars = [byte_ints_to_char_str(bits_to_byte_ints(cipher[i])) for i in split["train"]]
        src_tok = train_bpe_tokenizer(train_plain, vocab_size=args.vocab_src, use_whitespace_pretok=True)
        tgt_tok = train_bpe_tokenizer(train_cipher_chars, vocab_size=args.vocab_tgt, use_whitespace_pretok=False)
        print("source vocab size:", src_tok.get_vocab_size())
        print("target vocab size:", tgt_tok.get_vocab_size())

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_tok.save(str(out_dir / "src_tokenizer.json"))
        tgt_tok.save(str(out_dir / "tgt_tokenizer.json"))

        bad = 0
        for i in split["val"][:300]:
            char_str = byte_ints_to_char_str(bits_to_byte_ints(cipher[i]))
            ids = tgt_tok.encode(char_str).ids
            decoded = tgt_tok.decode(ids, skip_special_tokens=True)
            if decoded != char_str:
                bad += 1
        print(f"cipher tokenizer exact round-trip: {300 - bad}/300")
        assert bad == 0, "cipher tokenizer is lossy -- bit-level metrics would be meaningless"

    hf_username = None
    if args.push_to_hub:
        if "HF_TOKEN" not in os.environ:
            raise RuntimeError("--push-to-hub requires the HF_TOKEN environment variable to be set")
        from huggingface_hub import HfApi
        hf_username = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
        print("Hugging Face user:", hf_username)

    # Load any existing manifest first: running configs one at a time across
    # separate invocations (e.g. to fit a limited GPU session) must not clobber
    # already-completed configs' entries -- each invocation otherwise starts
    # from empty dicts and would overwrite LINKS.md/LINKS.json with only the
    # config(s) just run.
    results, links = {}, {}
    links_json_path = Path(args.output_dir) / "LINKS.json"
    if links_json_path.exists():
        prior = json.loads(links_json_path.read_text())
        results.update(prior.get("results", {}))
        links.update(prior.get("links", {}))

    for name in configs_to_run:
        run_config(name, plain, cipher, split, src_tok, tgt_tok, args, device, results, links, hf_username)

    print("\nRESULTS:", json.dumps(results, indent=2, default=str))
    print("\nLINKS:", json.dumps(links, indent=2, default=str))


if __name__ == "__main__":
    main()
