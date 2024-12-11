#!/bin/bash -l
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1                                                                                                                                                                    
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2                                                                                                                                                                
#SBATCH --cpus-per-gpu=8 
#SBATCH --job-name="demo"
#SBATCH --output="test.%j.out"
#SBATCH --account=garikipa_1359
#SBATCH --export=ALL
#SBATCH --exclude=a02-15
#SBATCH --open-mode=append
#SBATCH --mem=32G                                                                                                                                                                 

eval "$(conda shell.bash hook)"
conda activate MPP3
cd /project/garikipa_1359/rahulgul/spatiotemporal/MPP1/multiple_physics_pretraining/

export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=6

master_node=$SLURMD_NODENAME
VENVDIR=/project/garikipa_1359/rahulgul/spatiotemporal/MPP/
run_name="demo"
config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp                                                                              
yaml_config="./config/mpp_avit_b_config.yaml"


module purge
module load gcc/11.3.0 cmake/3.23.2 openmpi/4.1.4 emacs/28.2 git/2.36.1 python/3.11.3 hdf5/1.12.2 cuda/11.6.2 cudnn/8.4.0.27-11.6

source /project/garikipa_1359/rahulgul/spatiotemporal/MPP/environment/bin/activate
set -x

srun python `which torchrun` \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
                --rdzv_backend=c10d \
                --rdzv_endpoint=$master_node:29500 \
                train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config --use_ddp
