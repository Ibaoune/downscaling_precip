#!/bin/bash

#SBATCH --job-name=emul_cpu        # Job Name
#SBATCH --output=out_%j.log        # Everything (stdout & stderr) goes here
#SBATCH --error=out_%j.log         # Can also merge stderr with stdout
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --account=CLIMAT-UM6P-ST-IWRI-7KSIFKVWKUY-DEFAULT-CPU

########################################
# USER SHOULD SET THESE TO yes OR no
########################################
train="yes"        # yes= activate training flag / no= deactivate training flag
validation="yes"    # yes= activate validtion flag / no= deactivate validation flag
########################################



# Activate Conda environment
source ~/.bashrc
conda activate clean_env_Pytorch

# Record start time
start_time=$(date +%s)

# Run main script
# stdout and stderr are automatically redirected to the SLURM output file
#python3 -u train.py  # -u = unbuffered output for real-time logging

if [[ "$train" == "yes" ]]; then
    echo "Running training..."
    python3 -u train.py
fi

if [[ "$validation" == "yes" ]]; then
    echo "Running validation..."
    python3 -u eval.py
fi

if [[ "$train" != "yes" && "$validation" != "yes" ]]; then
    echo "Neither training nor validation selected. Set train=\"yes\" and/or validation=\"yes\"."
fi



# Record end time and log total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Job ${SLURM_JOB_ID} completed in $runtime seconds."

