#!/bin/sh 
#SBATCH --gres=gpu:L40:1
#SBATCH --partition=general
#SBATCH --mem=128Gb
#SBATCH --cpus-per-task=16
#SBATCH -t 3:20:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=eval_limit_1000_lora_iter_1
#SBATCH --error=/home/srijithr/iterative-alignment/lm-evaluation-harness/job_outputs/%x__%j.err
#SBATCH --output=/home/srijithr/iterative-alignment/lm-evaluation-harness/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu

source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate spin

export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export HF_TOKEN=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
export HYDRA_FULL_ERROR=1
cd /home/srijithr/iterative-alignment/lm-evaluation-harness

# WANDB no space betwenn equal sign
# python run_eval.py 735315
    # checkpoint_dir=/data/tir/projects/tir7/user_data/srijithr/spin_outputs/stage1_1024_lora_only_SPIN_iter0_L40_1e-5-3\
    # run_name=eval_limit_1000_lora_iter_0

# python run_eval.py 735835
#     checkpoint_dir='/data/tir/projects/tir7/user_data/srijithr/spin_outputs/stage1_1024_lora_only_SPIN_iter1_L40_1e-5_2'\
#     run_name=eval_limit_1000_lora_iter_1

# python run_eval.py \ 733369
#     checkpoint_dir='/data/tir/projects/tir7/user_data/srijithr/spin_outputs/stage1_1024_lora_only_SPIN_iter1_L40_1e-5_2' \
#     run_name=eval_limit_300_beta_1e-2_iter_0

# python run_eval.py \ 733370
#     checkpoint_dir='/data/tir/projects/tir7/user_data/srijithr/spin_outputs/stage1_1024_lora_only_SPIN_iter1_L40_1e-5_2' \
#     run_name=eval_limit_300_beta_1e-2_iter_1
#-------------------------------------------- 
# Changed batch size to 5 in below runs because base model 733391 OOMd with batch size 20

# python run_eval_model.py model_args=0 run_name=eval_limit_1000_repo_iter_0  # 735224
# python run_eval_model.py model_args=1 run_name=eval_limit_1000_repo_iter_1  735225
# python run_eval_model.py model_args=2 run_name=eval_limit_300_repo_base_model 733818