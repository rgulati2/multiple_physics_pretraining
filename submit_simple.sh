#!/bin/bash -l
#SBATCH --time=47:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name="1_cntxt16_fine"
#SBATCH --output="1_cntxt16_fine.%j.out"
#SBATCH --account=garikipa_1359
#SBATCH --export=ALL
#SBATCH --mem=32G
#SBATCH --open-mode=append

module list
nvidia-smi
source /project/garikipa_1359/rahulgul/spatiotemporal/2025/softwareInstall//pythonPath/bin/activate
python train_basic1.py --run_name "1_cntxt16_fine" --config "finetune"  --yaml_config "./config/mpp_avit_b_config.yaml"  
