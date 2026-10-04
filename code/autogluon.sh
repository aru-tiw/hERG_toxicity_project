#!/bin/bash
#SBATCH --job-name=autogluon
#SBATCH --account=bioinf595w26_class
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/artiwari/turbo_bioinf595/opt/Final_Project/autogluon_%j.log
#SBATCH --mail-user=artiwari@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/artiwari/turbo_bioinf595/opt/Final_Project

/nfs/turbo/dcmb-class/bioinf595/sec001/artiwari/opt/miniforge3/envs/autogluon_env/bin/python train_compare.py