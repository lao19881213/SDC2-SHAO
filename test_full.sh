#!/bin/bash

# sdc2

#SBATCH --job-name=full
#SBATCH --output=/o9000/SDC2/SHAO/full_out.txt
#SBATCH --error=/o9000/SDC2/SHAO/full_err.txt
#SBATCH --partition=purley-cpu
#SBATCH --time=200:00:00 
#SBATCH --nodes=3
#SBATCH --cpus-per-task=26
##SBATCH --ntasks-per-node=26
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000gb
#SBATCH --export=all
##SBATCH --exclude=purley-x86-cpu[02-08]

source /o9000/SDC2/SHAO/bashrc

start_time=`date +%s`

python /o9000/SDC2/SHAO/full.py

end_time=`date +%s`
duration=`echo "$end_time-$start_time" | bc -l`
echo "Total runtime = $duration sec"
