import os
import sys
import torch
import hydra
from pathlib import Path 
from omegaconf import DictConfig, OmegaConf


from datasets import Dataset, load_dataset
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from peft import LoraConfig
from accelerate import Accelerator


# Did not install trl from pip - instead using the local version to read thorugh code
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from trl.trl import  SFTTrainer ,SFTConfig
from utils import get_sft_dataset_alpaca, create_unique_dir_name

@hydra.main(config_path="./configs/", config_name="sft_config")
def run_code(cfg: DictConfig):
    set_seed(cfg.sft.seed)

    cfg.data.output_dir = create_unique_dir_name(cfg.data.output_dir)
    os.makedirs(str(cfg.data.output_dir), exist_ok=True)

    # Save the config file
    OmegaConf.save(cfg, os.path.join(cfg.data.output_dir, "config.yaml"))

    model = AutoModelForCausalLM.from_pretrained( # SRIJITH - You have to pass HF token as environment variable when launching jobs
        cfg.model.model_name,
        cache_dir = cfg.model.cache_dir,
        device_map = 'auto' ,
        #torch_dtype = torch.float16 / torch.bfloat16 ; Use this when you use the model for inference
        load_in_4bit=False) 
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_params = LoraConfig(
        lora_alpha = cfg.lora.lora_alpha,
        lora_dropout = cfg.lora.lora_dropout,
        r = cfg.lora.r, 
        bias = cfg.lora.bias,
        task_type = cfg.lora.task_type,
        target_modules = cfg.lora.target_modules.split(" "), # Have to do it this was as OmegaConf does not support lists
    )
    
    train_dataset = get_sft_dataset_alpaca( cfg.data.train_path , tokenizer)
    eval_dataset = train_dataset.filter(lambda x: len(x["text"])  <= 2000).select(range(400))

    training_params = SFTConfig(  # replaced TrainingArguments with SFTConfig 
        optim=cfg.sft.optim,  # e.g., "paged_adamw_32bit"
        warmup_steps=cfg.sft.warmup_steps,  # e.g., 100
        max_grad_norm=cfg.sft.max_grad_norm,  # e.g., 1.0
        weight_decay=cfg.sft.weight_decay,  # e.g., 0.001

        learning_rate=cfg.sft.learning_rate,  # e.g., 1e-4
        lr_scheduler_type=cfg.sft.lr_scheduler_type,  # e.g., "cosine"

        num_train_epochs=cfg.sft.num_train_epochs,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,  # e.g., 6
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,

        report_to=cfg.sft.report_to,  # e.g., "wandb"
        run_name=cfg.sft.run_name,  # e.g., 'debugging SFT with Alpaca data'

        logging_steps=cfg.sft.logging_steps,  # e.g., 10
        save_strategy=cfg.sft.save_strategy,  # e.g., 'epoch'
        eval_strategy=cfg.sft.evaluation_strategy,  # e.g., 'epoch'
        do_eval=cfg.sft.do_eval,  # e.g., True

        bf16=cfg.sft.bf16,  # e.g., True, adjust for GPU (e.g., A100 or 6000Ada)

        group_by_length=cfg.sft.group_by_length,  # e.g., True
        output_dir=cfg.data.output_dir,
        seed=cfg.sft.seed,
    )


    trainer = SFTTrainer(
        model=model,

        train_dataset=train_dataset,
        eval_dataset = eval_dataset,

        args=training_params,
        peft_config=peft_params,

        tokenizer=tokenizer,
        dataset_text_field='text',
        packing=False,
        max_seq_length=4000,
    )

    trainer.train()  
    trainer.model.save_pretrained(os.path.join(cfg.data.output_dir, "final_chekpoint"))

if __name__ == "__main__":
    run_code()