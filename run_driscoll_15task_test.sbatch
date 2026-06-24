#!/bin/bash
#SBATCH --partition=kempner
#SBATCH --account=kempner_dgc_lab
#SBATCH --job-name=driscoll15
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/driscoll15_%j.out
#SBATCH --error=logs/driscoll15_%j.err

cd ~/projects/flexible_multitask

module purge
module load python/3.10.13-fasrc01
module load cuda/11.8.0-fasrc01
module load cudnn/8.9.2.26_cuda11-fasrc01

source .venv/bin/activate

python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUS', tf.config.list_physical_devices('GPU'))"

python stepnet/general_model_train_15task_test.py
