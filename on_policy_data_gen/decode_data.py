# from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import datasets
import os
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
# import json
import jsonlines
import numpy as np
# import datasets
import sys
from config_generator import all_ref_model_names, all_ref_models
import copy
import random



parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--train_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help='Path to the LLM model')
parser.add_argument('--ref_model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--epoch', type=int, default=42,
                    help='epoch')
parser.add_argument('--seed', type=int, default=42,
                    help='seed')
parser.add_argument('--output_dir', type=str, default="datasets/llama3.1_8B_ultrafeedback",
                    help='output_dir')
parser.add_argument('--algo', type=str, default="alphaDPO",
                    help='Algo.')

if __name__ == "__main__":
    args = parser.parse_args()

    print(args)
    
    ref_model_name = all_ref_model_names[all_ref_models.index(args.ref_model)]
    
    output_file = f'all_train_data_{args.algo}_{args.epoch}.json'
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        sys.exit(0)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    ref_model = args.ref_model
    train_model = args.train_model


    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "./data/"))

    output_data = []

    dataset = load_from_disk(f"datasets/{ref_model_name}")
    for data in dataset:
        data['alpha'] = 0.0
        output_data.append(data)
        for _ in range(4):
            data = copy.deepcopy(data)
            data['alpha'] = 1.0
            chosen = []
            chosen.append({
                "role": "user",
                "content": data["prompt"]
            })
            chosen.append({
                "role": "assistant",
                "content": random.choice(data["all_generated_responses"])
            })
            rejected = []
            rejected.append({
                "role": "user",
                "content": data["prompt"]
            })
            rejected.append({
                "role": "assistant",
                "content": random.choice(data["all_generated_responses"])
            })
            data.update({
                "chosen": chosen,
                "rejected": rejected,
            })
            output_data.append(data)


    # output_file = f'all_train_data_{args.epoch}.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = datasets.Dataset.from_list(output_data)
    dataset = dataset.train_test_split(test_size=0.3)
    
    dataset.save_to_disk(os.path.join(args.output_dir, f"{ref_model_name}"))
    sys.exit(0)