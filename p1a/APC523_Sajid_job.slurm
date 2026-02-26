#!/bin/bash
#SBATCH --job-name=p1a_b   # job name
#SBATCH --nodes=1                    # run on a single node
#SBATCH --ntasks=1                   # run a single task
#SBATCH --cpus-per-task=1            # cpu cores per task
#SBATCH --mem=4G                     # total memory (4GB is plenty for N=64)
#SBATCH --time=00:10:00              # total run time limit (10 mins)
#SBATCH --mail-type=begin,end,fail   # send email when job begins, ends, or fails
#SBATCH --mail-user=rs8247@princeton.edu

# Load the environment
module purge
module load anaconda3/2025.6

# Run the python script
python p1abackup.py