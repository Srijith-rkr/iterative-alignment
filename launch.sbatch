#!/bin/sh 
#SBATCH --gres=gpu:L40:8
#SBATCH --partition=general
#SBATCH --mem=500Gb
#SBATCH --cpus-per-task=80
#SBATCH -t 0-16:00:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=geneal_8_gpu_wSAVECKPT_final_full_SFT_L40
#SBATCH --error=/home/srijithr/iterative-alignment/SPIN_implementation/job_outputs/%x__%j.err
#SBATCH --output=/home/srijithr/iterative-alignment/SPIN_implementation/job_outputs/%x__%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu
3
# THE 8 GPU run failed with CPU mem error with 256Gb, so trying 512Gb
# The long run started muuch earlier than general run 

source /data/tir/projects/tir7/user_data/srijithr/miniconda3/etc/profile.d/conda.sh
conda activate spin

export OMP_NUM_THREADS=10
# export DS_SKIP_CUDA_CHECK=1
export HF_DATASETS_CACHE=/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir
export HF_TOKEN=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

cd /home/srijithr/iterative-alignment/SPIN_implementation/SPIN

# accelerate launch --config_file configs/multi_gpu_debug_stage_1.yaml --num_processes=4 spin/run_spin.py configs/config_debug.yaml --run_name=beta_1e-2_stage1_1024_lora_only_SPIN_iter0_A100_80GB_1e-5 --learning_rate=1e-5 --beta=0.01

# 
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full.yaml --run_name=full_SFT_L40_8_27H

#  - long partition   737948
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=4 spin/run_spin.py configs/config_debug_full.yaml --run_name=4_BS_final_full_SFT_L40_4_54H

# SRIJITH _ I CHANGED GRADIENT ACCUMULATION STEPS TO 4 on gpu config 
# 739008
# accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=4 spin/run_spin.py configs/config_debug_full.yaml --run_name=4_BS_correctGA_final_full_SFT_L40_4_54H  --gradient_accumulation_steps=4 --per_device_train_batch_size=4

# SRIJITH _ I CHANGED GRADIENT ACCUMULATION STEPS TO 2 on gpu config: BS * did not work -> changed it back to 4
# 
accelerate launch --config_file configs/multi_gpu_debug_stage_3.yaml --num_processes=8 spin/run_spin.py configs/config_debug_full.yaml --run_name=geneal_8_gpu_wSAVECKPT_final_full_SFT_L40  --gradient_accumulation_steps=2 --per_device_train_batch_size=4




# /data/tir/projects/tir7/user_data/srijithr/miniconda3/condabin/conda /data/tir/projects/tir7/user_data/srijithr/miniconda3/lib/


# l40 is 128 only | happens next month only 
# a6000 is 64 and 128 - good here 
# 6000Ada is 64 only | 