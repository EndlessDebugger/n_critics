#!/bin/bash

# Directory where the log file will appear
LOG_DIR="./"  # change this to your actual log directory if needed

# Pattern to match (e.g., error_log.12345)
FILENAME_REGEX="^error_log\.[0-9]+$"

# remove previous job file 
rm error_log.*

# submit job 
sbatch jobfile.slurm

sleep 15 
FILE=./error_log.*
if test -f "$FILE"; then
tail -f -n30 $FILE
 


