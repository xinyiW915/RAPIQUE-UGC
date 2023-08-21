#!/bin/bash

#SBATCH --job-name=RAPIQUE
#SBATCH --partition gpu
#SBATCH --gres=gpu:0
#SBATCH --nodes=2
#SBATCH -o /mnt/storage/home/um20242/scratch/RVS-resize/log/2160P.out
#SBATCH --mem=50GB

cd "${SLURM_SUBMIT_DIR}"

echo Running on host "$(hostname)"
echo Time is "$(date)"
echo Directory is "$(pwd)"
echo Slurm job ID is "${SLURM_JOBID}"
echo This jobs runs on the following machines:
echo "${SLURM_JOB_NODELIST}"

module add apps/matlab/2018a
module load apps/ffmpeg/4.3

matlab -nodisplay -nodesktop -nosplash -singleCompThread < demo_compute_RAPIQUE_feats.m

