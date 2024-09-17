################ correct code ##################
import os
import sys
import argparse
import torch
import json

from datasets import Dataset, load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)

from peft import LoraConfig, PeftModel

from tqdm import tqdm
# from dpo_llama2_read_with_phi  import get_stack_exchange_paired

def get_stack_exchange_paired(file_path = '/home/srijithr/iterative-alignment/datasets/cherry_alpaca_5_percent_iteration_0.json'):
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset("json", data_files=file_path)['train'] 
    original_columns = dataset.column_names



    def return_prompt_and_responses(samples):
        return {
            "prompt": alpaca_prompt_template(samples),
            "chosen": [sample for sample in samples["output"]],
            "rejected": [sample for sample in samples["llama_generated_response"]],
        }

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

def alpaca_prompt_template(datapoint, include_response = False): # Changed the implementation to handel HF bached implementation; eg: input = [input1,input2,input3] 
    prompts = []

    for i in range(len(datapoint['input'])):
    
        if datapoint['input'][i] == '' or  'no input' in  datapoint['input'][i].lower() : 
            prompt =  f"""Below is an instruction that describes a task Write a response that appropriately completes the request.\n### Instruction:\n\n{datapoint['instruction'][i]}\n\n### Response:\n"""
        else :
            prompt =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{datapoint['instruction'][i]}\n\n### Input:\n{datapoint['input'][i]}\n\n### Response:\n"""

        if include_response : prompt = prompt + datapoint['output'][i]
        
        prompts.append(prompt)
    return prompts


def run_code():
    # Load base model
    # Load model directly
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
    import torch
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct", cache_dir ='/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir',dtype = torch.bfloat16)
    exit(0)
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", cache_dir ='/data/models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct/',dtype = torch.bfloat16)


    
    # model = AutoModelForCausalLM.from_pretrained(
    #     'meta-llama/Llama-2-7b-hf',
    #     device_map='auto',
    #     trust_remote_code=True,
    #     #cache_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/srijith-gpu/code/Users/t-srijithra/srijith_back_up/models/',
    # )#
    # do not know what this is used for
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1
    # model = PeftModel.from_pretrained(model, 'model_checkpoints/3kDPO_3e_1e-5_LLaMA2_FT_3e_1e-5_52K_dataset')
    model.eval()


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")#,cache_dir = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/srijith-gpu/code/Users/t-srijithra/srijith_back_up/models/',)# trust_remote_code=True)
    train_dataset = get_stack_exchange_paired()

    response_data = []
    for i in tqdm(range(len(train_dataset))): 
        prompt = train_dataset[i]['prompt']
        encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        greedy_output = model.generate(encoded_prompt,pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens= int(   1.25*len(  tokenizer.encode(train_dataset[i]['chosen'])  )  )  )
        greedy_output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        print(prompt)
        print(greedy_output[len(prompt):])
        print('-------------------------------------------------------------')

        response_data.append({
            **train_dataset[i],
            'model_response': greedy_output})

        if i %10 == 0:
            print(f'wrote {i} samples')

            with open('outputs/inferences.json','w') as fh:
                json.dump(response_data, fh, indent = 4) 

if __name__ == '__main__':
    run_code()

