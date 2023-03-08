import subprocess

CMD_CPU = 'sbatch -o /home/ubuntu/slurm.out /data/slurm/jobs/sysbench_cpu.script'
CMD_MEM = 'sbatch -o /home/ubuntu/slurm.out /data/slurm/jobs/sysbench_mem.script'
CMD_ML = 'sbatch -p test  -o /home/ubuntu/slurm.out /data/slurm/jobs/pytest.sh'

def execute(loadtype):
    if loadtype == 'cpu':
        # ret = subprocess.call(CMD_CPU, shell=True)
        ret = subprocess.check_output(["sbatch","-o","/home/ubuntu/slurm.out","/data/slurm/jobs/sysbench_cpu.script"])
    elif loadtype == 'mem':
        ret = subprocess.check_output(["sbatch","-o","/home/ubuntu/slurm.out","/data/slurm/jobs/sysbench_mem.script"])
        # ret = subprocess.call(CMD_MEM, shell=True)
    else:
        ret = subprocess.check_output(["sbatch","-p","test","-o","/home/ubuntu/slurm.out","/data/slurm/jobs/pytest.sh"])
        # ret = subprocess.call(CMD_ML, shell=True)
    return ret