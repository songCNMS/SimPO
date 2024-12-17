from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import os

# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
import jsonlines
import numpy as np
import datasets
import sys
from collections import defaultdict
import random
import copy
import torch
from config_generator import all_ref_model_names, all_ref_models


parser = argparse.ArgumentParser(description="Decode with vllm")
parser.add_argument(
    "--ref_model",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Path to the LLM model",
)
parser.add_argument(
    "--temperature", type=float, default=1.2, help="Temperature for sampling"
)
parser.add_argument(
    "--top_p", type=float, default=0.95, help="Top-p probability for sampling"
)
parser.add_argument(
    "--max_tokens", type=int, default=4096, help="Maximum number of tokens to generate"
)

parser.add_argument("--num_samples", type=int, default=1000, help="num_samples")
parser.add_argument("--epoch", type=int, default=42, help="epoch")
parser.add_argument("--seed", type=int, default=42, help="seed")
parser.add_argument(
    "--output_dir",
    type=str,
    default="datasets/llama3.1_8B_ultrafeedback",
    help="output_dir",
)



if __name__ == "__main__":
    args = parser.parse_args()
    
    ref_model = args.ref_model
    
    ref_model_name = all_ref_model_names[all_ref_models.index(ref_model)]
    data_dir_loc = os.getenv("AMLT_OUTPUT_DIR", "./data/")
    output_file = f"{data_dir_loc}/mh/all_train_data_{ref_model_name}_{args.seed}.json"
    data_dir_loc = os.path.join(os.getenv("AMLT_DATA_DIR", "data/"))

    d_prompts = []
    dbar_prompts = []
    with jsonlines.open(f"{data_dir_loc}/data/all_train_data.json") as reader:
        for obj in reader:
            if obj["type"] == "D":
                d_prompts.append(obj["prompt"])
            else:
                dbar_prompts.append(obj["prompt"])

    d_prompts = sorted(list(set(d_prompts)))
    dbar_prompts = sorted(list(set(dbar_prompts)))
    # if len(dbar_prompts) > len(d_prompts):
    #     dbar_prompts = np.random.choice(
    #         dbar_prompts, size=int(len(d_prompts)), replace=False
    #     )

    all_prompts = list(d_prompts) + list(dbar_prompts)
    prompt_resp_dict = {}
    
    
    ref_llm = LLM(model=ref_model, tensor_parallel_size=torch.cuda.device_count())
    ref_tokenizer = ref_llm.get_tokenizer()
    conversations = [
        ref_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in all_prompts
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        # temperature=i/2.0,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        logprobs=10,
        seed=args.seed,
    )
    
    outputs = ref_llm.generate(conversations, sampling_params)
    for j, output in enumerate(outputs):
        prompt_resp_dict[all_prompts[j]] = (output.outputs[0].cumulative_logprob, output.outputs[0].text)

    del ref_llm
    del ref_tokenizer

    output_data = []
    with jsonlines.open(f"{data_dir_loc}/data/all_train_data.json") as reader:
        for obj in reader:
            if obj["prompt"] in prompt_resp_dict:
                obj["model_response"] = prompt_resp_dict[obj["prompt"]][1]
                obj["model_response_logprob"] = prompt_resp_dict[obj["prompt"]][0]
                output_data.append(obj)
            

    with jsonlines.open(output_file, "w") as writter:
        writter.write_all(output_data)
