module purge
module load python/3.8.10
module load scipy-stack

echo "loading module done"

echo "Creating new virtualenv"

virtualenv --no-download ~/$1
source ~/$1/bin/activate

echo "Virtual environment creation done"

pip install --no-index --upgrade pip

pip install torch==1.10.0+computecanada torchvision==0.11.1+computecanada
pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
pip install mmdet
pip install wandb
pip install numpy
pip install packaging
pip install pillow

echo " Installing packages done"