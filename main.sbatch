#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --account=PAS3162
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=results_%j.txt

module load python/3.10
module load cuda/11.8.0

cd /users/PAS3162/yang6294/cv_project

python main.py --m all