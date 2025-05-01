#!/bin/sh 
#SBATCH --gres=gpu:a100:1              	# Can request 1-4 t4, 1-2 a100, or 1-2 rtx
#SBATCH --partition=gpu              	# Request the GPU partition/queue
#SBATCH --time=12:00:00 		        # Set the max runtime, in wallclock hours
#SBATCH --job-name=door_apps_run	# Job name
#SBATCH --mem=80G 			# Memory allocation per node 
#SBATCH --output=std.%j			# Output file names for stdout and stderr
#SBATCH --ntasks=2 			# Total number of tasks/CPU cores (for HPRC recommended method, this is = # CPUs
#SBATCH --ntasks-per-node=2		# Tasks (CPUs) per node
#SBATCH --output=error_log.%j
#Total number of compute nodes = # CPUs requested / CPUs per node
#So 4 CPUs requested divided by 4 CPUs per node = 1 compute node. 

# Load modules
module purge 
module restore proj2

# check for GPU access
nvidia-smi

# CD to project dir
cd $SCRATCH/n_critics/daniel

echo "Activating venv..." 
source activate_venv proj2

echo "Starting apps run..." 
python3 make_apps.py 
echo "apps run complete."

deactivate

