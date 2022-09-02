#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=16
#SBATCH --tasks-per-node=16
#SBATCH --constraint=haswell
#SBATCH --sdn


number_of_workers=256
module load python3/3.9-anaconda-2021.11


export OMP_NUM_THREADS=8
echo We have nodes: ${SLURM_JOB_NODELIST}

echo "$SDN_IP_ADDR"

hn=$(hostname -s)
port="8786"
echo ${port}
echo "starting scheduler"
dask-scheduler --no-dashboard --no-bokeh --no-show --host ${hn} --port ${port} &
echo "starting workers"
srun -o dask_worker_info.txt dask-worker ${hn}:${port} &
echo "starting gp2Scale"
python -u run_gp2Scale.py ${hn}:${port} ${number_of_workers}
