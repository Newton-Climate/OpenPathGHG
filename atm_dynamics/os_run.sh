#!/usr/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=test_job.%j.out
#SBATCH --error=test_job.%j.err
#SBATCH --time=24:00:00
#SBATCH -p normal
#SBATCH -c 20
#SBATCH --mem=10GB
module load python/3.6.1
module load py-numpy/1.19.2_py36
python3 OS.py