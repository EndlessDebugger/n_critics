#!/bin/sh 
#SBATCH --gres=gpu:a100:1              	# Can request 1-4 t4, 1-2 a100, or 1-2 rtx
#SBATCH --partition=gpu              	# Request the GPU partition/queue
#SBATCH --time=00:40:00		        # Set the max runtime, in wallclock hours
#SBATCH --job-name=generate_image_job	# Job name
#SBATCH --mem=150G 			# Memory allocation per node 
#SBATCH --ntasks=24 			# Total number of tasks/CPU cores (for HPRC recommended method, this is = # CPUs
#SBATCH --ntasks-per-node=24		# Tasks (CPUs) per node
#SBATCH --output=error_log.%j		# output filenames for stdout and stderr
#Total number of compute nodes = # CPUs requested / CPUs per node
#So 4 CPUs requested divided by 4 CPUs per node = 1 compute node. 

# Load modules
module purge 
module restore python_3_10_8_grace

# check for GPU access
nvidia-smi

# CD to project dir
cd $SCRATCH/n_critics/logan

echo "Activating venv..." 
source ../../.venv/bin/activate

echo "Starting N-critics..." 
python3 n_critics.py 
echo "N-critics complete."

deactivate

