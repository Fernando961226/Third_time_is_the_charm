# How to train on compute canada
## This folder contains two main subfolder:
--submit_scripts/ <br>
--output_logs/

The folder `submit_scripts` contains following shell scripts: 
1. train_detector.sh -- The job submission script to train a detector with the specified resource
2. check_directory.sh -- check if the environment named `mmdet` exists or not  
3. create_virtualenv.sh -- creates a virtual environment
4. train_detector_loop.sh -- Takes the `config_file_path` and `wandbRUNNAME` as a cmd line argument (to be used with submit_loop.sh)
5. submit_loop.sh -- To submit a sequence of runs in a loop

The folder `output_logs` contains the output files generated by a job submitted in compute canada.

## How to use? 

### 1. create_virtualenv.sh  

1. To create a new environment, `cd` to the folder `job_submission_computecanada_/submit_scripts/
2. run `bash create_virtualenv.sh <envname>` this will create a new environment called envname in the root folder
The contents of the file is as follows:

```sh
module purge # get rid of modules loaded before (this is not neccesary to use but is a good practice)
module load python/3.8.10 # load the python module
module load scipy-stack # load the scipy stack which contains modules like numpy scipy

echo "loading module done"

echo "Creating new virtualenv"

virtualenv --no-download ~/$1 # The $1 is a cmd line variable, in shell scripting the $1 represents the first variable passed to the script, $2 is second variable and so on 
source ~/$1/bin/activate # This is the procedure to activate the environment when using virtualenv

echo "Virtual environment creation done"

pip install --no-index --upgrade pip  # upgrade pip

## list all your pip packages here
pip install torch==1.10.0+computecanada torchvision==0.11.1+computecanada
pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install mmdet
pip install wandb
pip install numpy
pip install packaging
pip install pillow
echo " Installing packages done"
```
## 2. train_detector.sh 

1. write all the resource requeirement as show below
2. run `sbatch train_detector.sh`
```sh
#!/bin/bash
#SBATCH --nodes 1 -- This means request one node
#SBATCH --gpus-per-node=v100l:4 -- request 4 GPU of type v100
#SBATCH --tasks-per-node=4 -- Should be equal to no. of GPU
#SBATCH --cpus-per-task=6 -- No. of CPU, will be useful in the dataloader part
#SBATCH --mem=100G -- The CPU memory
#SBATCH --time=8:00:00 -- The time to request the resources 
#SBATCH --output=/home/m32patel/projects/def-dclausi/share/whale/mmwhale/job_submission_computecanada_/output_logs4/%j.out -- The file where the output will be stored. If you look closely, the filename is given to be %j.out which nothing but the jobid of your job
#SBATCH --account=def-dclausi -- The name of the sponsor
#SBATCH --mail-user=muhammed.computecanada@gmail.com -- To get the email notification of the job timelines
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

===================================================================================
The above lines will get you the resources or hardware which you want to use

In the below lines specify what you want to do with the hardware you were allocated
===================================================================================
module purge -- Get rid of modules which were loaded before on

module load python/3.8.10 -- load python 
module load scipy-stack --- load scipy stack


echo "Loading module done"

source ~/mmdet4/bin/activate -- Activate the environment named mmdet4

echo "Activating virtual environment done"

#cd /project/def-dclausi/share/whale/mmwhale/
cd /project/6075102/share/whale/mmwhale/  -- go to the project folder

echo "starting training..."
srun python -u tools/train.py configs/custom_configs/whale_sahi/retinanet_r50_fpn_1x_whale_sahi_coco_256_256.py --launcher="slurm" --wandbRunName=$1

==================================================================================
There are so many things to keep note of in the above command

1. srun should be used when you are using multi-gpu
2. tool/train.py is your python script
3. configs/custom_configs/whale_sahi/retinanet_r50_fpn_1x_whale_sahi_coco_256_256.py is the mmdet config file
4. --launcher="slurm" mmdetection has inbuilt support to run script which uses SLURM based HPC(COMPUTE CANADA is slurm based HPC). So you can use multi-gpu, multi-node training easily with just 1 line and mmdet will take care of all the dirty stuff for you
5. wandbRunName is the runname which will appear in wandb dashboard
==================================================================================
```

### 3. submit loop.sh

```sh
======================================================================
Place the config filepath in the array and the wandbRunName in array 2
======================================================================
array=(
# "configs/custom_configs/whale_sahi/faster_rcnn_r50_fpn_512.py"   
"configs/custom_configs/whale_sahi/retinanet_r50_fpn_512.py"
# "configs/custom_configs/whale_sahi/faster_rcnn_r50_fpn_256.py"
# "configs/custom_configs/whale_sahi/retinanet_rcnn_r50_fpn_256.py"
)

array2=(
# "faster_rcnn_r50_fpn_512.py"
"retinanet_r50_fpn_512.py"
# "faster_rcnn_r50_fpn_256.py"
# "retinanet_rcnn_r50_fpn_256.py"
)

for i in "${!array[@]}"; do
   sbatch train_detector_loop.sh ${array[i]} ${array2[i]}
   echo "task successfully submitted"
   sleep 10
done
=================================================================================================
The above will loop through array1 and array2 simulateneously and submit the job using sbatch 

One thing to note here is when youre submitting job sequentially, put a sleep stamtement 

(that's how compute canada works!)
====================================================================================================

```
