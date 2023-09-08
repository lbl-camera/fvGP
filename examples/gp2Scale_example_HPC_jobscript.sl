#!/bin/bash
#SBATCH -A m4055_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -n 32
#SBATCH --ntasks-per-node=4
#SBATCH -c 32   ##### 2 * [64/ntasks-per-node]
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3


export SLURM_CPU_BIND="cores"
number_of_workers=32

source /global/homes/m/mcn/gp2Scale/gp2Scale_env/bin/activate

export OMP_NUM_THREADS=8
echo We have nodes: ${SLURM_JOB_NODELIST}

echo "$SDN_IP_ADDR"

hn=$(hostname -s)
port="8786"
echo ${port}
echo "starting scheduler"
dask-scheduler --no-dashboard --no-show --host ${hn} --port ${port} &
echo "starting workers"
srun -o dask_worker_info.txt dask-worker ${hn}:${port} &
echo "starting gp2Scale"
python -u run_gp2ScaleGPU.py ${hn}:${port} ${number_of_workers}

