import os
from datasets import load_dataset
from typing import Dict, Optional

def create_unique_dir_name(base_dir):
    # If base directory does not exist, return it
    if not os.path.exists(base_dir):
        return base_dir
    else:
        # Find the next available directory name with a suffix
        counter = 2
        new_dir = f"{base_dir}_{counter}"
        while os.path.exists(new_dir):
            counter += 1
            new_dir = f"{base_dir}-{counter}"
        return new_dir

def alpaca_prompt_template(datapoint, tokenizer, include_response = False):
        check = datapoint['input'].lower()
        if datapoint['input'] == "" or ('no' in check and 'input' in check) or ('none' in check): 
            
            prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
### Instruction:\n{datapoint['instruction']}\n\n### Response:\n"""

        else:
            prompt =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
### Instruction:\n{datapoint['instruction']}\n\n### Input:\n{datapoint['input']}\n\n### Response:\n"""


        if include_response : prompt = prompt + datapoint['output'] + tokenizer.eos_token
        return prompt

def get_sft_dataset_alpaca(path,tokenizer):

    dataset = load_dataset("json", data_files=path)['train'] 
    original_columns = dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "text": alpaca_prompt_template(samples, tokenizer, include_response=True)}

    return dataset.map(
        return_prompt_and_responses,
        batched=False,
        remove_columns=original_columns,
    )

def get_spin_sft_train_dataset(data_file, inference= False):

    dataset = load_dataset("json", data_files=data_file)['train'] 
    original_columns = dataset.column_names

    def alpaca_prompt_template(datapoint,inference): # Changed the implementation to handel HF bached implementation; eg: input = [input1,input2,input3] 
        prompts = []

        for i in range(len(datapoint['real'])):
            prompt =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions..\n\n"
            prompt = prompt + f"{datapoint['real'][i][0]['role']}: {datapoint['real'][i][0]['content']}\n\n{datapoint['real'][i][1]['role']}: "
            if not inference:
                prompt = prompt + datapoint['real'][i][1]['content']
            prompts.append(prompt)
            
        return prompts

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        return {
            "text": alpaca_prompt_template(samples,inference)}

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )



def get_SPIN_train_dataset(data_file, inference= False, use_key = None, spin_template= False, tokenizer=None):

    dataset = load_dataset("json", data_files=data_file)['train'] 
    original_columns = dataset.column_names

    def dpo_prompt_template(datapoint): # Changed the implementation to handel HF bached implementation; eg: input = [input1,input2,input3] 
        prompts = []

        for i in range(len(datapoint['real'])):
            prompt =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions..\n\n"
            prompt = prompt + f"{datapoint['real'][i][0]['role']}: {datapoint['real'][i][0]['content']}\n\n{datapoint['real'][i][1]['role']}: "
            prompts.append(prompt)
            
        return prompts

    def return_dpo_prompt_and_responses(samples) -> Dict[str, str]:
        if use_key:
            chosen = [sample + tokenizer.eos_token for sample in samples[use_key]]
        else :
            chosen = [sample[1]['content'] + tokenizer.eos_token for sample in samples["real"]]
            
        if spin_template :
            
            return {
                "prompt": dpo_prompt_template(samples),
                "real": chosen,
                "generated": [sample[1]['content'] + tokenizer.eos_token  for sample in samples["generated"]],
            }
            
        else:
            
            return {
                "prompt": dpo_prompt_template(samples),
                "chosen": chosen,
                "rejected": [sample[1]['content'] + tokenizer.eos_token  for sample in samples["generated"]],
            }

        

    return dataset.map(
        return_dpo_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )



def get_alpaca_dpo_train_dataset(data_file, use_key = None, tokenizer=None, swap = False):

    dataset = load_dataset("json", data_files=data_file)['train'] 
    original_columns = dataset.column_names

    def alpaca_prompt_template(datapoint, include_response = False):
        check = datapoint['input'].lower()
        if datapoint['input'] == "" or ('no' in check and 'input' in check) or ('none' in check): 
            
            prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
### Instruction:\n{datapoint['instruction']}\n\n### Response:\n"""

        else:
            prompt =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
### Instruction:\n{datapoint['instruction']}\n\n### Input:\n{datapoint['input']}\n\n### Response:\n"""

        if include_response : prompt = prompt + datapoint['output'] + tokenizer.eos_token
        return prompt

    def return_dpo_prompt_and_responses(samples) -> Dict[str, str]:
        # if use_key:
        #     chosen = [sample + tokenizer.eos_token for sample in samples[use_key]]
        # else :
        #     chosen = [sample['output']+ tokenizer.eos_token for sample in samples["real"]]
        if swap: 
            print('swapped')
            return {
                "prompt": alpaca_prompt_template(samples),
                "chosen": samples[use_key] + tokenizer.eos_token, 
                "rejected": samples['output'] +  tokenizer.eos_token,
            }
        else :
        
            return {
                "prompt": alpaca_prompt_template(samples),
                "chosen": samples['output'] +  tokenizer.eos_token,
                "rejected":  samples[use_key] + tokenizer.eos_token,
            }

        

    return dataset.map(
        return_dpo_prompt_and_responses,
        batched=False,
        remove_columns=original_columns,
    )



def get_spin_kto_train_dataset(data_file, inference= False,use_key = None, tokenizer=None):

    dataset = load_dataset("json", data_files=data_file)['train'] 
    original_columns = dataset.column_names

    def dpo_prompt_template(datapoint): # Changed the implementation to handel HF bached implementation; eg: input = [input1,input2,input3] 
        prompts = []

        for i in range(len(datapoint['real'])):
            prompt =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions..\n\n"
            prompt = prompt + f"{datapoint['real'][i][0]['role']}: {datapoint['real'][i][0]['content']}\n\n{datapoint['real'][i][1]['role']}: "
            prompts.append(prompt)
            
        return prompts

    def return_dpo_prompt_and_responses(samples) -> Dict[str, str]:
        prompts = dpo_prompt_template(samples)
        if use_key:
            chosen_completions =[sample + tokenizer.eos_token for sample in samples[use_key]]
            rejected_completions = [sample + tokenizer.eos_token for sample in samples[use_key]]
        else:
            chosen_completions =[sample[1]['content'] + tokenizer.eos_token for sample in samples["real"]]
            rejected_completions = [sample[1]['content'] + tokenizer.eos_token for sample in samples["generated"]]
        chosen_label = [True for sample in samples["real"]]
        rejected_label = [False for sample in samples["real"]]
        
        # prompts.extend(prompts)
        # completions = chosen_completions.extend(rejected_completions)
        # labels = chosen_label.extend(rejected_label)
        # for i in samples:
        #     completions.append(i['real'][i][0])
        #     [sample['real'][1]['content'] for sample in samples["real"]]
        
        #completion_rejected = [sample[1]['content'] for sample in samples["generated"]]
        
        prompts.extend(prompts)
        chosen_completions.extend(rejected_completions)
        chosen_label.extend(rejected_label)
        
        # Handeling cases where completion is a null string | Or else the HF dataprocessing backend throws errors 
        for i in range(len(chosen_completions)):
            if chosen_completions[i] == '':
                chosen_completions[i] = ' '
        
        return {
            "prompt":  prompts,
            "completion":chosen_completions,
            "label": chosen_label,
        }

        

    return dataset.map(
        return_dpo_prompt_and_responses,
        batched=True,
        remove_columns=original_columns,
    )

# def get_spin_dpo_train_dataset(data_file, inference= False, use_key = None):

#     dataset = load_dataset("json", data_files=data_file)['train'] 
#     original_columns = dataset.column_names

#     def dpo_prompt_template(datapoint): # Changed the implementation to handel HF bached implementation; eg: input = [input1,input2,input3] 
#         prompts = []

#         for i in range(len(datapoint['real'])):
#             prompt =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions..\n\n"
#             prompt = prompt + f"{datapoint['real'][i][0]['role']}: {datapoint['real'][i][0]['content']}\n\n{datapoint['real'][i][1]['role']}: "
#             prompts.append(prompt)
            
#         return prompts

#     def return_dpo_prompt_and_responses(samples) -> Dict[str, str]:
#         if use_key:
#             chosen = [sample for sample in samples[use_key]]
#         else :
#             chosen = [sample[1]['content']  for sample in samples["real"]]
        
            
#         return {
#             "prompt": dpo_prompt_template(samples),
#             "chosen": chosen,
#             "rejected": [sample[1]['content'] for sample in samples["generated"]],
#         }

        

#     return dataset.map(
#         return_dpo_prompt_and_responses,
#         batched=True,
#         remove_columns=original_columns,
#     )


# def spin_prompt_template_for_alpaca_eval(datapoint, include_response = False):
    
#     prompt =  "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions..\n\n"
#     prompt = prompt + f"user: {datapoint['instruction']}\n\nassistant: "
            
#     if include_response : prompt = prompt + datapoint['output']

#     return prompt



# train_dataset = get_spin_sft_train_dataset(data_file="/home/azureuser/Srijith_workspace/Finetune-RCA/scoring/data/5K_complexity_difference_X_quality_difference_spin0.json")
# eval_dataset = train_dataset.filter(lambda x: len(x["text"])  <= 2000).select(range(200))

# train_dataset = get_spin_dpo_train_dataset(data_file="/home/azureuser/Srijith_workspace/Finetune-RCA/scoring/data/5K_complexity_difference_X_quality_difference_spin0.json")
# eval_dataset = train_dataset.filter(lambda x:len(x["prompt"]) + len(x['chosen'])  <= 2000).select(range(200))

# train_dataset = get_spin_dpo_train_dataset(data_file="hf_alignment/temp_data_for_jobs/5K_iter1_full_quality_difference.json",use_key = "5K_full_quality_difference_deita_spin0_LLaMA2_DPO_1e-5_0.1B_3e")

# print('for bp')
# TO CHECK PROMPT LENGTH 
# lens = []
# for i in full_dataset['text']:
#     lens.append(len(i))
# lens = np.array(lens)
# print(np.histogram(lens, bins=[0,1000,2000,3000,4000,5000,6000,7000,50000]))


# from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
# tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
# # # train_dataset = get_SPIN_train_dataset(data_file="/home/azureuser/Srijith_workspace/Finetune-RCA/hf_alignment/temp_data_for_jobs/5K_iter1_full_quality_difference.json",
# # #                                                use_key='5K_full_quality_difference_deita_spin0_LLaMA2_DPO_1e-5_0.1B_3e',
# # #                                                tokenizer=tokenizer)
# # # train_dataset = get_spin_kto_train_dataset(data_file="/home/azureuser/Srijith_workspace/Finetune-RCA/hf_alignment/temp_data_for_jobs/5K_iter1_full_quality_difference.json",
# # #                                                use_key='5K_full_quality_difference_deita_spin0_LLaMA2_DPO_1e-5_0.1B_3e',
# # #                                                tokenizer=tokenizer)
# # # get_spin_kto_train_dataset(data_file="5K_full_quality_difference_spin0.json")

# train_dataset = get_alpaca_dpo_train_dataset(data_file="/home/azureuser/Srijith_workspace/Finetune-RCA/hf_alignment/temp_data_for_jobs/alpaca_poc/3k_quality_difference_LLaMA2_FT_3e_1e-4_52K_dataset_3e_iter1.json",
#                                             use_key='LLaMA2_FT_3e_1e-4_52K_dataset_3e',
#                                             tokenizer=tokenizer,swap=True)
# print('dsf')

