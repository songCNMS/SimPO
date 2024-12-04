#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed
import subprocess
import datasets


from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from simpo_trainer import SimPOTrainer
from simpo_config import SimPOConfig
from dataclasses import dataclass, field
from typing import Optional, Literal
import jsonlines
import os
from dpo_trainer import AlphaDPOTrainer


logger = logging.getLogger(__name__)

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "simpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if tokenizer.bos_token and example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if tokenizer.bos_token and example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main(ep=1):
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = parser.parse()
    
    # training_args.output_dir = training_args.output_dir + f"_{ep}"
    data_dir = list(data_args.dataset_mixer.keys())[0]
    data_args.dataset_mixer = {f"{data_dir}/{training_args.trainer_type}_dataset_{ep}": 1.0}
    ref_model = model_args.model_name_or_path
    model_args.model_name_or_path = training_args.output_dir + f"/{training_args.trainer_type}_{ep}"
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
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data parameters {data_args}")
    

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args.dataset_mixer,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "type", "alpha"],
        # seed=training_args.seed,
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names.remove("type")
    column_names.remove("alpha")

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    
    if ref_model.lower().find("qwen") >= 0:
        if tokenizer.eos_token is not None:
            tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
            tokenizer.bos_token_id = tokenizer.eos_token_id


    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "simpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    
    
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
        
    eval_d_list = []
    eval_dbar_list = []
    for item in raw_datasets["test"]:
        if item["type"] == "D":
            eval_d_list.append(item)
        else:
            eval_dbar_list.append(item)
            
    eval_d_dataset = datasets.Dataset.from_list(eval_d_list)
    eval_dbar_dataset = datasets.Dataset.from_list(eval_dbar_list)

    model = model_args.model_name_or_path
    
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )
    
    print(eval_d_dataset, eval_dbar_dataset)
    
    training_args.model_init_kwargs = model_kwargs
    #########################
    # Instantiate SimPO trainer
    #########################
    # if training_args.trainer_type == "simpo":
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["test"],
        eval_dataset=eval_d_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    # else:
    # training_args.ref_model_init_kwargs = model_kwargs
    # trainer = AlphaDPOTrainer(
    #         model=model,
    #         ref_model=ref_model,
    #         args=training_args,
    #         train_dataset=raw_datasets["test"],
    #         eval_dataset=eval_d_dataset,
    #         tokenizer=tokenizer,
    #         peft_config=get_peft_config(model_args),
    #         max_length=training_args.max_length,
    #         max_prompt_length=training_args.max_prompt_length,
    #         loss_type=training_args.loss_type,
    #     )
    

    ##########
    # Evaluate
    ##########
    logger.info("*** Evaluate D***")
    eval_d_metrics = trainer.evaluate()
    eval_d_metrics["eval_samples"] = len(eval_d_dataset)
    
    
    # if training_args.trainer_type == "simpo":
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["test"],
        eval_dataset=eval_dbar_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )
    # else:
    # training_args.ref_model_init_kwargs = model_kwargs
    # trainer = AlphaDPOTrainer(
    #         model=model,
    #         ref_model=ref_model,
    #         args=training_args,
    #         train_dataset=raw_datasets["test"],
    #         eval_dataset=eval_dbar_dataset,
    #         tokenizer=tokenizer,
    #         peft_config=get_peft_config(model_args),
    #         max_length=training_args.max_length,
    #         max_prompt_length=training_args.max_prompt_length,
    #         loss_type=training_args.loss_type,
    #     )

    ##########
    # Evaluate
    ##########
    logger.info("*** Evaluate DBAR***")
    eval_dbar_metrics = trainer.evaluate()
    eval_dbar_metrics["eval_samples"] = len(eval_dbar_dataset)
    
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)

    # logger.info("*** Training complete! ***")
    return eval_d_metrics, eval_dbar_metrics


from omegaconf import OmegaConf
import sys
from datetime import datetime
import pandas as pd


if __name__ == "__main__":
    cfg = OmegaConf.from_cli()
    ep = cfg.epoch
    exp_name = cfg.exp_name
    eval_d_metrics, eval_dbar_metrics = main(ep=ep)
    eval_d_metrics["exp_name"] = f"D_{exp_name}_{ep}"
    eval_dbar_metrics["exp_name"] = f"DBAR_{exp_name}_{ep}"
    run_name = cfg.get("run_name", datetime.today().strftime("%Y%m%d%H%M%S"))
    
    output_dir_loc = os.path.join(os.getenv('AMLT_OUTPUT_DIR', f"./logs/{run_name}/"))
    os.makedirs(output_dir_loc, exist_ok=True)
    with jsonlines.open(f"{output_dir_loc}/metrics.jsonl", "a") as writter:
            writter.write(eval_d_metrics)
            writter.write(eval_dbar_metrics)
            
            
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{output_dir_loc}/running.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    if os.path.exists(f"{output_dir_loc}/metrics.csv"):
        df_ex = pd.read_csv(f"{output_dir_loc}/metrics.csv")
        metrics_monitor = df_ex.columns.tolist()
    else:
        df_ex = None
        metrics_monitor = ["exp_name"] + [x for x in eval_d_metrics.keys() if x.startswith("eval_rewards") or x.startswith("eval_logps")]
    
    data = [[eval_d_metrics.get(x, None) for x in metrics_monitor], [eval_dbar_metrics.get(x, None) for x in metrics_monitor]]
    df = pd.DataFrame(data=data, columns=metrics_monitor)
    if df_ex is not None:
        df = pd.concat([df_ex, df], axis=0)
    
    df.to_csv(f"{output_dir_loc}/metrics.csv", index=False)
    sys.exit(0)