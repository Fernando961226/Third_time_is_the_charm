
#

# HOW TO USE THIS FILE:

# Place the config path relative to the mmcls folder in the array 1
# Place the wandb run namer in the array2
# run this file from the jobsubmission/submit_scripts/ folder

#

# array=(
# "configs/custom_configs/resnet_50_kirtan_64_01.py"   
# "configs/custom_configs/resnet_50_kirtan_64_005.py"
# "configs/custom_configs/resnet_50_kirtan_64_0025.py"
# "configs/custom_configs/resnet_50_kirtan_128_01.py"
# "configs/custom_configs/resnet_50_kirtan_128_005.py"
# "configs/custom_configs/resnet_50_kirtan_128_0025.py")

# array2=(
# "resnet_50_kirtan_64_01.py"
# "resnet_50_kirtan_64_005.py"
# "resnet_50_kirtan_64_0025.py"
# "resnet_50_kirtan_128_01.py"
# "resnet_50_kirtan_128_005.py"
# "resnet_50_kirtan_128_0025.py")



array=(
"configs/custom_configs/whale_sahi/faster_rcnn_r50_fpn_1024.py"   
"configs/custom_configs/whale_sahi/retinanet_r50_fpn_512.py"
"configs/custom_configs/whale_sahi/retinanet_r50_fpn_1024.py"

# "configs/custom_configs/whale_sahi/retinanet_rcnn_r50_fpn_256.py"
)

array2=(
"faster_rcnn_r50_fpn_1024.py"
"retinanet_r50_fpn_512.py"
"retinanet_r50_fpn_1024.py"
# "retinanet_rcnn_r50_fpn_256.py"
)




# for val1 in ${list[*]}; do
#      echo $val1
# done


for i in "${!array[@]}"; do
   # bash test_echo.sh ${array[i]} ${array2[i]}
   sbatch train_detector_loop.sh ${array[i]} ${array2[i]}
   echo "task successfully submitted"
   sleep 10

done


# bash job_submission_computecanada/submit_scripts/test_echo.sh variable_name1 print_this
