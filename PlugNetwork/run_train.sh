#!/bin/bash
#SBATCH --job-name=train
#SBATCH -A research
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-type=END
#SBATCH --mail-user=praguna.manvi@research.iiit.ac.in
#SBATCH -N 1    

conda activate plg_env

echo 'running code'
python3 train.py --epoch 5
echo 'run code'

