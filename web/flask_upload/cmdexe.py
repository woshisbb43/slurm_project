import subprocess

CMD_CPU = 'sbatch -o /home/ubuntu/slurm.out /data/slurm/jobs/sysbench_cpu.script'
CMD_MEM = 'sbatch -o /home/ubuntu/slurm.out /data/slurm/jobs/sysbench_mem.script'
CMD_ML = 'sbatch -p test  -o /home/ubuntu/slurm.out /data/slurm/jobs/pytest.sh'

def execute(loadtype):
    if loadtype == 'cpu':
        ret = subprocess.call(CMD_CPU, shell=True)
    elif loadtype == 'mem':
        ret = subprocess.call(CMD_MEM, shell=True)
    else:
        ret = subprocess.call(CMD_ML, shell=True)
    if ret == 0:
        return 'success'
    else:
        return 'failed'