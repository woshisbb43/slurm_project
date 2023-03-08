#!/bin/bash
SBATCH --job-name=myjob
SBATCH --output=myjob.out
SBATCH --error=myjob.err
SBATCH --ntasks=1
SBATCH --time=10:00
SBATCH --mem=20M

echo "Hello, World!" >> /tmp/result.txt
