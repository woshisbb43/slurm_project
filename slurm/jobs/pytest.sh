#!/bin/bash
#SBATCH --job-name=ml-training
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00

module load python/3.6
cd /data/ml
sudo python3 /data/ml/image_train.py