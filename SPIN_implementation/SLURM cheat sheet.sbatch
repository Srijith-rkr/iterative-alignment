#!/bin/sh 
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=debug
#SBATCH --mem=32Gb
#SBATCH -t 00:20:00              # time limit:  add - for days (D-HH:MM) 
#SBATCH --job-name=building_env
#SBATCH --error=/home/srijithr/%j.err
#SBATCH --output=/home/srijithr/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=srijithr@andrew.cmu.edu

TO LAUNCH INTERACTIVE JOB 
srun --gres=gpu:L40:1  --cpus-per-task=2 --partition=general --job-name=debugging_code  --mail-type=ALL --mail-user=srijithr@andrew.cmu.edu --wait=0 -t 1:59:10 -n 1 --pty bash
--gres=gpu:6000Ada:1 
--gres=gpu:A6000:1 
# gpu:2080Ti:3


DATA LOCATION 
/data/tir/projects/tir7/user_data/srijithr

PYTHON path 
/data/tir/projects/tir7/user_data/srijithr/miniconda3/envs/dpo/bin/python3.10 


USEFUL SINFO FLAGS
sinfo --format="%P %D %C %G" 
Partition, Num_Nodes, CPU, GPU_TYPE
--states=idle
--partition=debug
| sort | uniq - prints all unique GPUs
| grep "gpu:A6000"

My frequent command to check available GPUs  : sinfo --format="%P %D %C %G %N" --states=idle

To check left over time :  scontrol show job 555534 | grep Run

OTHER USEFUL COMMANDS
sstat: Resource utilization by a particular job
sacct: View the history of your recently run jobs; Did they complete? fail?

TO_COMPARE_GPUs
https://bizon-tech.com/gpu-benchmarks/NVIDIA-Quadro-RTX-8000-vs-NVIDIA-A100-40-GB-(PCIe)-vs-NVIDIA-RTX-6000-Ada/532vs592vs640


Common sinfo Format Options: 
sinfo --helpformat

	•	%all: Print all available fields.
	•	%P: Partition name.
	•	%D: Number of nodes.
	•	%C: CPU information in the form A/I/O/T (Allocated/Idle/Other/Total).
	•	%G: Generic resources (GRES), which include GPU information.
	•	%N: Node list (names of the nodes).
	•	%t: Node state (e.g., idle, allocated).
	•	%E: Reason for node state if not available (e.g., drained, down).
	•	%m: Memory size of the node.
	•	%O: CPU overload (number of CPUs overallocated).
	•	%T: Total time limit for the partition.
	•	%c: Number of CPUs per node.
	•	%f: Features (e.g., specific hardware properties of the node).
	•	%l: Time limits for the partition.
	•	%b: Number of GPUs allocated.
	•	%r: Number of running jobs on the node.