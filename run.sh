#!/bin/bash
#SBATCH -p gpu-train
#SBATCH -J llm
#SBATCH --nodes 1
#SBATCH --gres=gpu:l40:2
#SBATCH --ntasks-per-node 2
#SBATCH --mem 128gb
#SBATCH -c 4
#SBATCH --time=7-0
#SBATCH --output=logs/llm_%J.log
#SBATCH --error=logs/llm_%J.log
#SBATCH --exclude=gpu73,gpu74,gpu79,gpu76

# For multi-gpu running
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # e.g. gpu=72
echo "MASTER_ADDR="$MASTER_ADDR
export MASTER_PORT=22698 ###USE A DIFFERENT MASTER PORT
export NCCL_P2P_DISABLE=1
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_NODELIST"
echo "Number of GPUs per node: $SLURM_GPUS_ON_NODE"

# Regular args:
script_name=$1
if [ -z "$script_name" ]; then
    echo "Using default script arg"
    script_name=experiments.main
fi

script_full_path="scripts.$script_name"

./stop_script.sh $1

# Log file
logs_dir="logs"
log_file_name=${script_name//./_}
log_file="$logs_dir/$log_file_name.log"
mkdir -p $logs_dir

echo "Starting: $script_full_path"

## JB: See llm_loading.py for loading of llama and other models.
LLAMA_DIR=/net/scratch/jbutch/nlp/models
# img=/home/jbutch/Projects/HT25/memprofiler/container.sif
export LLAMA_DIR=$LLAMA_DIR

# Activate your conda environment
export PYTHONPATH=/home/jbutch/Projects/HT25/icl_repro:$PYTHONPATH
source /software/conda/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate icl
srun --kill-on-bad-exit python -u -m $script_full_path

# nohup python -u -m $script_full_path > $log_file 2>&1 &
# apptainer exec --nv --bind /home/jbutch:/home/jbutch $img python -u -m $script_full_path


echo "Done. Exiting."