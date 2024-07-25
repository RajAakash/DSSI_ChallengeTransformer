#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.txt
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH -A bdx
#SBATCH --partition=pbatch
#SBATCH --time=00:20:00

python3 Transformer.py