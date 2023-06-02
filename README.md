# What's the output of this project?
Please visit 
# What's this repo for?
This repo is for presenting how to use ansible provision a slurm compute nodes cluster.
# Structure of this repo?
## ansible
Ansible folder contains neccessary files (playbook, templates, etc.) to provision the slurm compute nodes cluster
## slurm jobs
Demo jobs to provide workloads. Currently 4 jobs: ML job, cpu job, memory job and a test script.
## web
A web service which has 3 main func:
1. Predict a flower image
2. Trigger a pre-defined slurm job
3. A simple monitor tooling on current slurm cluster
