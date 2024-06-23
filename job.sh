#!/bin/bash
# Grid Engine options (lines prefixed with #$)
#$ -N dataprocessing

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1

#$ -cwd
#$ -l h_rt=05:10:00
#$ -l h_vmem=200G

#$ -m bea -M s2550585@ed.ac.uk

# Initialise the environment modules
. /etc/profile.d/modules.sh
module load cuda

# Load Python
module load anaconda # this loads a specific version of anaconda
conda activate /exports/eddie/scratch/s2550585/anaconda/envs/mypython # this starts the environment

# Run the program


#python /exports/eddie/scratch/s2550585/dissertation/dataset/process_amazon_2023.py --domain Baby_Products
python /exports/eddie/scratch/s2550585/dissertation/dataset/process_amazon_2023.py --domain Video_Games
python /exports/eddie/scratch/s2550585/dissertation/dataset/process_amazon_2023.py --domain Amazon_Fashion

