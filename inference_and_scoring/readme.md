This folder contains:
* code to  score the quality and complexity of samples using the open scource code for the DEITA paper (https://arxiv.org/abs/2312.15685) 
* Run multi-gpu inference to generate samples at each iteration 

`deita`: 
* Modified implementation of https://github.com/hkust-nlp/deita to enable multi-gpu scoring with Hugging Face accelerate 

`multi_gpu_complexity_scorer_alpaca.py | multi_gpu_quality_scorer_local_spin.py`: 
* Used to generate quality score for the prompt, prompt + chosen_response, prompt + rejected_response 

`multi_gpu_quality_scorer_local_alpaca.py | multi_gpu_complexity_scorer_spin.py`: 
* Used to generate complexity score for the prompt, prompt + chosen_response, prompt + rejected_response 

* uses accelerate launch to run scripts 
* `'accelerate launch --num_processes 8 abc.py'`