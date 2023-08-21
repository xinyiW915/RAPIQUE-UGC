#!/bin/bash

#SBATCH --job-name=regreugc
#SBATCH --partition=cpu
#SBATCH --nodes=6
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH -o /mnt/storage/home/um20242/scratch/RAPIQUE-UGC/logs/regression_ugc.out
#SBATCH --mem-per-cpu=10G

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

module load languages/anaconda3/2020.02-tflow-1.15
#conda create -n reproducibleresearch pip python=3.6

# Activate virtualenv
source activate reproducibleresearch
#pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run Python script
python < evaluate_bvqa_features_regression.py
#nohup python -u evaluate_bvqa_features_regression.py>360.log 2>&1 &

## Deactivate virtualenv
conda deactivate