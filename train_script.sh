# Get the absolute path of the script directory automatically
# If it does not work, specify ProjDir manually
ProjDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Project directory: $ProjDir"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $ProjDir/stdiff/configs/accelerate_config.yaml --num_processes 1 --main_process_port 29580 $ProjDir/stdiff/train_stdiff.py --train_config ./configs/kth_train_config.yaml
