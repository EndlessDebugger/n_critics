#!/bin/sh 
#SBATCH --gres=gpu:a100:1              	# Can request 1-4 t4, 1-2 a100, or 1-2 rtx
#SBATCH --partition=gpu              	# Request the GPU partition/queue
#SBATCH --time=10:00:00 		        # Set the max runtime, in wallclock hours
#SBATCH --job-name=generate_image_job	# Job name
#SBATCH --mem=200G 			# Memory allocation per node 
#SBATCH --output=std.%j			# Output file names for stdout and stderr
#SBATCH --ntasks=32 			# Total number of tasks/CPU cores (for HPRC recommended method, this is = # CPUs
#SBATCH --ntasks-per-node=32		# Tasks (CPUs) per node
#SBATCH --output=error_log.%j
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

