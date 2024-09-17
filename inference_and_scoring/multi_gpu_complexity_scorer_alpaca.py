import os
import sys
import json
import hydra
import numpy as np
from tqdm import tqdm 
from pathlib import Path 
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf

from accelerate import Accelerator
from accelerate.utils import gather_object, tqdm

wd = Path(__file__).parent
sys.path.append(str(wd / 'deita' / 'src'))
sys.path.append(str(wd.parent / 'training'))
from deita.selection.scorer  import Llama_Scorer
from utils import create_unique_dir_name



@hydra.main(config_path="./", config_name="cmplx_scorer_config")
def run_code(cfg: DictConfig):

    cfg.data.output_dir = create_unique_dir_name(cfg.data.output_dir)
    os.mkdir(cfg.data.output_dir)
    accelerator = Accelerator()

    model_name_or_path = "hkust-nlp/deita-complexity-scorer"

    scorer = Llama_Scorer(model_name_or_path, accelerator, cache_dir = cfg.model.cache_dir)
    scorer.model.eval()

    def alpaca_prompt_template(datapoint, include_response = False):
        check = datapoint['input'].lower()
        if datapoint['input'] == "" or ('no' in check and 'input' in check) or ('none' in check): 
            
            prompt =  f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n
    ### Instruction:\n{datapoint['instruction']}\n\n### Response:\n"""

        else:
            prompt =  f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
    ### Instruction:\n{datapoint['instruction']}\n\n### Input:\n{datapoint['input']}\n\n### Response:\n"""


        if include_response : prompt = prompt + datapoint['output']
        return prompt
        
    with open(cfg.data.train_path,'r') as fh: 
        dataset = json.load(fh)

    accelerator.wait_for_everyone()  

    def get_score(sample):
        input_text = alpaca_prompt_template(sample) #sample[0]['content']
        #output_text = sample[1]['content']
        return scorer.infer_complexity(input_text)

    def get_full_score(sample, use_model_output = False, use_key = False):
        input_text = alpaca_prompt_template(sample)
        if use_model_output:
            output_text = sample['output']
        elif use_key:
            output_text = sample[use_key]
        return scorer.infer_full_complexity(input_text, output_text)
        
    with accelerator.split_between_processes(dataset) as prompts:
        to_json = []
        score_list = []
        difference_list = []
        missed_samples = 0
        missed_json = []
        for n, datapoint in enumerate(tqdm(prompts)):
            # Handeling Chosen Responses 
            try: 
                prompt_score = get_score(datapoint)
                chosen_score =  get_full_score(datapoint,use_model_output = True)
                # rejected_score = get_full_score(datapoint, use_key = 'LLaMA2_FT_3e_1e-4_52K_dataset_3e')
            except Exception as e:
                print(f"Skipping sample coz of {e}")
                missed_json.append({**datapoint}); missed_samples+=1
                with open(f'outputs/GPU_{accelerator.process_index}_MISSED_spin_iter_0_complexity_score.json', 'w') as fh: # SRIJITH
                    json.dump(missed_json, fh, indent = 4)
                continue
            
            
            to_json.append({ **datapoint,
                            'prompt_complexity_score':prompt_score,
                            'chosen_complexity_score':chosen_score,
                            # 'rejected_complexity_score':rejected_score
                            })
            # difference_list.append(chosen_score-rejected_score)
            score_list.append(prompt_score)
            
            
            if n % 100 == 0 or n ==  (len(dataset) - 1):
                
                print('\nPROMPT COMPLEXITY')
                counts, bin_edges = np.histogram(score_list, bins=5)
                for i in range(len(counts)):
                    print(f'from {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {counts[i]} ') 
                    
                print('\nFULL COMPLEXITY DIFFERENCE')
                counts, bin_edges = np.histogram(difference_list, bins=5)
                for i in range(len(counts)):
                    print(f'from {bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}: {counts[i]} ')
                    
                print(f'{(np.array(difference_list)<0).sum()} / {n} are < 0')
                print(f'missed {missed_samples} samples')
                
                with open(f'{cfg.data.output_dir}/gpu_{accelerator.process_index}_{cfg.data.dataset_note}_complexity_score.json', 'w') as fh: 
                    json.dump(to_json, fh, indent = 4)
                    
    results_gathered=gather_object(to_json)
    missed_samples_gathered=gather_object(missed_json)

    if accelerator.is_main_process:
        print('********************')
        print(f"Writing combined results of len {len(results_gathered)}")
        print('********************')
        with open('{cfg.data.output_dir}/final.json','w') as fh:
            json.dump(results_gathered, fh, indent = 4) 
            
        with open('{cfg.data.output_dir}/misssed_samples_final.json','w') as fh:
            json.dump(missed_samples_gathered, fh, indent = 4)


if __name__ == '__main__':
    run_code()
                
            
