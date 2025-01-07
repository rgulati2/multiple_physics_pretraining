#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --job-name="demo"
#SBATCH --output="MPP.%j.out"
#SBATCH --account=garikipa_1359
#SBATCH --export=ALL
#SBATCH --mem=16G
#SBATCH --open-mode=append

cd /project/garikipa_1359/rahulgul/spatiotemporal/MPP1/multiple_physics_pretraining/
conda activate MPP3
module list

nvidia-smi
source /project/garikipa_1359/rahulgul/spatiotemporal/MPP/environment/bin/activate

python train_basic.py --run_name "demo" --config "basic_config"  --yaml_config "./config/mpp_avit_b_config.yaml"  
