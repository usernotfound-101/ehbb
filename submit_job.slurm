#!/bin/bash
#SBATCH --job-name=ayush_anlp
#SBATCH --partition=u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs_%x.log

# Launches cipher_transformer.py on Ada. Assumes setup_env.sh has already been
# run once (so .venv/ exists) from a node with internet.
#
# Defaults to --wandb-mode offline: on this partition's 2080 Ti nodes,
# wandb.init() has been observed to fail with a CommError (can't reach
# api.wandb.ai from the compute node) while huggingface_hub's API was
# reachable -- so --push-to-hub is left on, but W&B logs locally instead.
# cipher_transformer.py also now catches a wandb.init() failure at runtime
# and continues training without it rather than crashing, so even if you
# switch back to --wandb-mode online and it fails again, the job won't die --
# but offline avoids wasting the retry/timeout on every config and gives you
# an actual local run to sync later:
#   sbatch submit_job.slurm                  # trains C1_base, 40 epochs (defaults)
#   sbatch submit_job.slurm C3_gqa            # one specific config
#   sbatch submit_job.slurm all 40            # all five configs, sequentially, one GPU
#
# After the job: from the login node (or wherever wandb.ai IS reachable),
#   cd <this dir> && wandb sync wandb/offline-run-*
# uploads the logged runs so you get real W&B URLs for the report/README.
#
# NOTE on --output=logs_%x.log: %x is the job name, not the job ID, so
# submitting more than one job with the same --job-name overwrites the same
# log file (SLURM truncates on each job start, it doesn't append). If you
# submit multiple configs as separate jobs, either override the name per
# submission (`sbatch --job-name=ayush_anlp_C3 submit_job.slurm C3_gqa`) or
# rely on this script's own outputs_<config>/ directory + wandb for the
# actual per-config record instead of this log file.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# module load python/3.10 cuda/12.1   # uncomment if Ada needs an explicit
                                       # module for python3/CUDA visibility
source .venv/bin/activate

# Credentials: never hardcode these here or commit them. Keep them in a
# local, private file and source it, e.g. once:
#   printf 'export WANDB_API_KEY=...\nexport HF_TOKEN=...\n' > ~/.anlp_secrets
#   chmod 600 ~/.anlp_secrets
if [ -f "$HOME/.anlp_secrets" ]; then
    source "$HOME/.anlp_secrets"
fi

CONFIG="${1:-C1_base}"
EPOCHS="${2:-40}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname)  config=${CONFIG}  epochs=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

python cipher_transformer.py \
    --config "${CONFIG}" \
    --epochs "${EPOCHS}" \
    --output-dir "outputs_${CONFIG}" \
    --wandb-mode offline \
    --push-to-hub

echo "done: ${CONFIG}"
