#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100l:1 # request a GPU
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=/home/m32patel/projects/def-dclausi/share/whale/mmwhale/job_submission_computecanada_/output_logs4/%j.out
#SBATCH --account=def-dclausi
#SBATCH --mail-user=muhammed.computecanada@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

module purge

module load python/3.8.10
module load scipy-stack


echo "Loading module done"

source ~/mmdet4/bin/activate

echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd /project/6075102/share/whale/mmwhale/

echo "starting training..."
#python tools/train.py configs/custom_configs/coco/retinanet_r50_fpn_1x_coco_coco.py --wandbRunName=$1 # mscoco
# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times
#python tools/train.py configs/custom_configs/whale/retinanet_r50_fpn_1x_coco_whale.py --wandbRunName=$1 # whale
# python tools/train.py configs/custom_configs/whale_sahi/faster_rcnn_r50_fpn_1x_whale_sahi_coco.py --wandbRunName=$1 --options "runner.max_epochs=2"
python -u tools/train.py $1 --wandbRunName=$2
