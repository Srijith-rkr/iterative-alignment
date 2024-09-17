import os
import sys
import torch
import hydra
from pathlib import Path 
from typing import Dict, Optional
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field


from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from peft import LoraConfig, PeftModel
from accelerate import Accelerator, PartialState

# Did not install trl from pip - instead using the local version to read thorugh code
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from trl.trl import  DPOConfig, DPOTrainer
from utils import get_SPIN_train_dataset, get_alpaca_dpo_train_dataset, create_unique_dir_name

@hydra.main(config_path="./configs/", config_name="dpo_config")
def run_code(cfg: DictConfig):
    set_seed(cfg.dpo.seed)

    cfg.data.output_dir = create_unique_dir_name(cfg.data.output_dir)
    os.makedirs(str(cfg.data.output_dir), exist_ok=True)

    # Save the config file
    OmegaConf.save(cfg, os.path.join(cfg.data.output_dir, "config.yaml"))

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        cache_dir = cfg.model.cache_dir,
        device_map= 'auto' ,
        #{"": PartialState().process_index}, #from https://github.com/huggingface/trl/issues/1220
       # device_map={"": Accelerator().local_process_index},
        #torch_dtype = torch.float16 / torch.bfloat16 ; Use this when you use the model for inference
    )
        # metadata={
        #     "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
        #     "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        # },
    
    # if script_args.ignore_bias_buffers:
    #     # torch distributed hack
    #     model._ddp_params_and_buffers_to_ignore = [
    #         name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
    #     ]

    if cfg.merge.do_merge:
        model = PeftModel.from_pretrained(model, cfg.merge.adapter_path)
        model = model.merge_and_unload()
        print(f'Merged this ckpt {cfg.merge.adapter_path}')


    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True) 
    tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer.padding_side)

    peft_params = LoraConfig(
        lora_alpha = cfg.lora.lora_alpha,
        lora_dropout = cfg.lora.lora_dropout,
        r = cfg.lora.r, 
        bias = cfg.lora.bias,
        task_type = cfg.lora.task_type,
        target_modules = cfg.lora.target_modules.split(" "), # Have to do it this was as OmegaConf does not support lists
    )


    train_dataset = get_alpaca_dpo_train_dataset(data_file=cfg.data.train_path,
                                               use_key=cfg.data.use_key,
                                               tokenizer=tokenizer,
                                               swap = True).select(range(40))
    
    eval_dataset = train_dataset.filter(lambda x:len(x["prompt"]) + len(x['chosen'])  <= 2000).select(range(4))


    training_args = DPOConfig(
        optim=cfg.dpo.optimizer_type,  # e.g., "paged_adamw_32bit"
        warmup_steps=cfg.dpo.warmup_steps,  # e.g., 100
        max_grad_norm=cfg.dpo.max_grad_norm,  # e.g., 1.0
        # weight_decay=cfg.dpo.weight_decay, Not implemented in old implementation

        learning_rate=cfg.dpo.learning_rate,  # e.g., 1e-4
        lr_scheduler_type=cfg.dpo.lr_scheduler_type,  # e.g., "cosine"

        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,  # e.g., 6
        per_device_eval_batch_size=cfg.dpo.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,

        report_to=cfg.dpo.report_to,  # e.g., "wandb"
        run_name=cfg.dpo.run_name,  # e.g., 'debugging SFT with Alpaca data'

        logging_steps=cfg.dpo.logging_steps,  # e.g., 10
        save_strategy=cfg.dpo.save_strategy,  # e.g., 'epoch'
        eval_strategy=cfg.dpo.evaluation_strategy,  # e.g., 'epoch'
        do_eval=cfg.dpo.do_eval,  # e.g., True

        bf16=cfg.dpo.bf16,  # e.g., True, adjust for GPU (e.g., A100 or 6000Ada)

        output_dir=cfg.data.output_dir,
        seed=cfg.dpo.seed,
        gradient_checkpointing_kwargs=dict(use_reentrant=cfg.dpo.gradient_checkpointing_use_reentrant),
        gradient_checkpointing=cfg.dpo.gradient_checkpointing,
        remove_unused_columns=False,
    )


    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,

        train_dataset=train_dataset,
        eval_dataset = eval_dataset,

        beta=cfg.dpo.beta, # SRIJITH
        args=training_args,
        peft_config=peft_params,

        tokenizer=tokenizer,
        max_prompt_length=cfg.dpo.max_prompt_length,
        max_length=cfg.dpo.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(os.path.join(cfg.data.output_dir, "final_chekpoint"))

    # 7. save
    dpo_trainer.model.save_pretrained(os.path.join(cfg.data.output_dir, "final_chekpoint"))


if __name__ == "__main__":
    run_code()