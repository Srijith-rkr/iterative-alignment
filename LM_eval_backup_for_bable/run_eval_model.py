import os
import lm_eval
import json
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lm_eval.loggers import WandbLogger
from lm_eval.models.huggingface import HFLM 
import time

from transformers import AutoModelForCausalLM, set_seed, TrainingArguments
from peft import PeftConfig, PeftModel,LoraConfig, get_peft_model
from safetensors import safe_open
from lm_eval.tasks import TaskManager

@hydra.main(config_path=".", config_name="eval_config")
def run_code(cfg: DictConfig):

    run_name = str(cfg.run_name)
    # checkpoint_dir = str(cfg.checkpoint_dir)
    output_dir = os.path.join(str(cfg.output_dir), run_name)
    model_args_list = ["pretrained=UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0,dtype=bfloat16"]
    model_args_list.append("pretrained=UCLA-AGI/zephyr-7b-sft-full-SPIN-iter1,dtype=bfloat16")
    model_args_list.append("pretrained=alignment-handbook/zephyr-7b-sft-full,revision=ac6e600eefcce74f5e8bae1035d4f66019e93190,dtype=bfloat16")
    
    model_args = int(cfg.model_args)
    model_args = model_args_list[model_args]
    print(OmegaConf.to_container(cfg, resolve=True))

    os.makedirs(output_dir, exist_ok=True)
   
    # if not os.path.exists(checkpoint_dir):
    #     raise FileNotFoundError(f"Error: The file {checkpoint_dir} does not exist.")



    # tensors = {} 
    # with safe_open("/data/tir/projects/tir7/user_data/srijithr/spin_outputs/-31/checkpoint-2334/adapter_model.safetensors", framework="pt", device=0) as f:
    #     for k in f.keys():
    #         tensors[k] = f.get_tensor(k)
        
    # model = AutoModelForCausalLM.from_pretrained('alignment-handbook/zephyr-7b-sft-full',
    #                                             cache_dir = '/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir',
    #                                             device_map = 'auto',
    #                                             torch_dtype = torch.bfloat16,
    #                                             revision =  'ac6e600eefcce74f5e8bae1035d4f66019e93190',
    #                                             trust_remote_code= False,
    #                                             use_flash_attention_2 = False,)
    # peft_config = LoraConfig(
    #         r=adapter_config['r'],
    #         lora_alpha=adapter_config['lora_alpha'],
    #         lora_dropout=adapter_config['lora_dropout'],
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #         target_modules=adapter_config['target_modules'],
    #         # modules_to_save=adapter_config['modules_to_save'],
    #     )
    # model = get_peft_model(model, peft_config) # Erroring here 
    # if isinstance(model, PeftModel):
    #     print('Model is an instance of PeftModel')
        
    # if model_args.adapter_path: inside spin trainer
    #         model = PeftModel.from_pretrained(model, model_args.adapter_path)
    #         model = model.merge_and_unload()
    #         print(f'Merged this ckpt {model_args.adapter_path}')

    # model = PeftModel.from_pretrained(model, checkpoint_dir) 
    # model = model.merge_and_unload()
    # model = HFLM(model)
    # # model_args = "pretrained=UCLA-AGI/zephyr-7b-sft-full-SPIN-iter0,dtype=bfloat16"
    # # model_args = "pretrained=UCLA-AGI/zephyr-7b-sft-full-SPIN-iter1,dtype=bfloat16"
    # model_args = "pretrained=alignment-handbook/zephyr-7b-sft-full,revision=ac6e600eefcce74f5e8bae1035d4f66019e93190,dtype=bfloat16"
    # UCLA-AGI/SPIN_iter1
    
    

# pretrained='UCLA-AGI/SPIN_iter0',dtype='bf16'
# #revision=step100000,dtype="bf16" \

    task_manager = TaskManager() # checks if you are using current tasks and popluates self with task information if you pass it through cli
    evals_to_run = {  'arc_challenge':{'few_shot': 25},
                    'truthfulqa':{'few_shot': 0},
                    'winogrande':{'few_shot': 5},
                    'gsm8k':{'few_shot': 5},
                    'hellaswag':{'few_shot': 10},
                    'mmlu':{'few_shot': 5},}

    # Use this to double_check task_names = task_manager.match_tasks(task_list)

    # if len(task_names) != len(task_list):
    #     raise ValueError(f"Error: The following tasks were not found: {set(task_list) - set(task_names)}")


    collect_all_results = {'to_check_model_args_later':model_args}
    start = time.time()
    for eval in evals_to_run:

        results = lm_eval.simple_evaluate(
            model='hf',
            model_args = model_args,
            tasks=eval,
            log_samples=True,
            num_fewshot=evals_to_run[eval]['few_shot'],
            # trust_remote_code=False,
            batch_size=5,
            device="cuda", # "cuda:0",
            # output_path='/home/srijithr/iterative-alignment/lm-evaluation-harness/outputs',
            task_manager= task_manager,
            random_seed=0,
            limit=1000,
            # Set below to match simple_evaluate() call from CLI
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
        end = time.time()

        results['config']['model_dtype'] = str(results['config']['model_dtype'])
        collect_all_results[eval] = results['results']
        
        with open(os.path.join(output_dir, f'{eval}.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    collect_all_results['config'] = OmegaConf.to_container(cfg, resolve=True)
    with open(os.path.join(output_dir, f'Table.json'), 'w') as f:
            json.dump(collect_all_results, f, indent=4)
            
    print(f"Time taken for {eval} is {end-start} seconds")

if __name__ == '__main__':
    run_code()
    

