import re
import os 
import sys
import torch
import hydra
import logging
from pathlib import Path


from transformers import AutoModelForCausalLM, set_seed
import transformers
from peft import PeftConfig, PeftModel
from accelerate import Accelerator
from torch.utils.data import Subset

# Did not install trl from pip - instead using the local version to read thorugh code
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import trl.trl 

from trl.trl import  DPOConfig, DPOTrainer
from trl.trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length

from SPIN.spin.alignment import (
    DataArguments,
    SPINConfig,
    H4ArgumentParser,
    ModelArguments,

    SPINTrainer,

    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from utils import get_SPIN_train_dataset, create_unique_dir_name

logger = logging.getLogger(__name__)

def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SPINConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Increase distributed timeout to 3h to enable push to Hub to complete
    accelerator = Accelerator()

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    # tokenizer.pad_token = tokenizer.eos_token
    
    
    ###############
    # Load datasets
    ###############
    train_dataset = get_SPIN_train_dataset(    data_file=data_args.data_path,
                                               use_key = data_args.use_key,
                                               spin_template=True,
                                               tokenizer = tokenizer)
    eval_dataset = train_dataset.filter(lambda x:len(x["prompt"]) + len(x['real'])  <= 2000).select(range(200))
    # SRIJITH Simply replace below line with your data loading method - but put infenrcne sample in generated 
    # raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    # logger.info(
    #     f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # column_names = list(raw_datasets["train"].features)

    #####################
    # Apply chat template
    #####################
    # raw_datasets = raw_datasets.map(
    #     apply_chat_template,
    #     fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     desc="Formatting comparisons with prompt template",
    # )

    # Replace column names with what TRL needs, text_real -> real and text_generated -> generated
    # for split in ["train", "test"]:
    #     raw_datasets[split] = raw_datasets[split].rename_columns(
    #         {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"}
    #     )

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map= 'auto', #get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        cache_dir=model_args.cache_dir,
    )

    model = model_args.model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    
    if model_args.adapter_path:
        model = PeftModel.from_pretrained(model, model_args.adapter_path)
        model = model.merge_and_unload()
        print(f'Merged this ckpt {model_args.adapter_path}')

    #########################
    # Instantiate spin trainer
    #########################
    spin_trainer = SPINTrainer(
        model,
        ref_model=None, 
       # model_init_kwargs=model_kwargs,
      #  ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        padding_value =  tokenizer.eos_token_id # Was defaulting to 0 and DPO used this 
    )

    ###############
    # Training loop
    ###############
    train_result = spin_trainer.train()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
    
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    spin_trainer.log_metrics("train", metrics)
    spin_trainer.save_metrics("train", metrics)
    spin_trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    training_args.output_dir = create_unique_dir_name(training_args.output_dir)
    spin_trainer.save_model(os.path.join(training_args.output_dir,'final_checkpoint'))
    # Save everything else on main process
    if accelerator.is_main_process:
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        spin_trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        spin_trainer.model.config.use_cache = True
        spin_trainer.model.config.save_pretrained(os.path.join(training_args.output_dir,'final_checkpoint'))

    # Ensure we don't timeout on model save / push to Hub
    logger.info("*** Waiting for all processes to finish ***")
    accelerator.wait_for_everyone()

    logger.info("*** Run complete! ***")


if __name__ == "__main__":
    main()
