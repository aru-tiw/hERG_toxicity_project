#!/bin/bash
#SBATCH --job-name=umap_search
#SBATCH --account=bioinf595w26_class
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/artiwari/turbo_bioinf595/opt/Final_Project/umap_search_%j.log
#SBATCH --mail-user=artiwari@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

cd /home/artiwari/turbo_bioinf595/opt/Final_Project

/nfs/turbo/dcmb-class/bioinf595/sec001/artiwari/opt/miniforge3/envs/project_env/bin/python umap_hyperparam_search.py