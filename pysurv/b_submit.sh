#!/bin/bash
timestamp=$(date +%m%d_%H%M)
cd /mnt/lustre/helios-home/nuttepet/ml-unos2/pysurv
output_dir=output/$timestamp
mkdir -p $output_dir
job_script="$output_dir/pysurv_$timestamp.pbs"
# Generate the job script with the timestamp in the job name
cat > $job_script << EOL
#!/bin/bash
### Job Name
#PBS -N job_$timestamp

### required runtime
#PBS -l walltime=24:00:00

### queue for submission
#PBS -q cpu_b

### Merge output and error files
#PBS -j oe

### Request memory and CPU cores on the compute node
#PBS -l select=1:mem=370G:ncpus=16

### start job in the directory it was submitted from
cd \$PBS_O_WORKDIR


### activate the Python virtual environment (if applicable)
source /mnt/lustre/helios-home/nuttepet/anaconda3/etc/profile.d/conda.sh

conda activate pysurv

### run the application
python main.py --timestamp $timestamp > $output_dir/output.log 2>&1
EOL
# Submit the job script to PBS
qsub $job_script