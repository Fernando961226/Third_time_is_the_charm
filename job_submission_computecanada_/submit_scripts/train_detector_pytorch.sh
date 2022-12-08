#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:2 # request a GPU
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=/home/m32patel/projects/def-dclausi/share/whale/mmwhale/job_submission_computecanada_/output_logs4/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE




CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


module purge

module load python/3.8.10
module load scipy-stack


echo "Loading module done"

source ~/mmdet5/bin/activate

echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd /project/6075102/share/whale/mmwhale/


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=$GPUS \
#     --master_port=$PORT \
#     tools/train.py \
#     $CONFIG \
#     --seed 0 \
#     --launcher pytorch ${@:3}

srun python -m torch.distributed.launch \
    tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}