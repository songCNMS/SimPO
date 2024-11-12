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
    
    output_file = f'all_train_data_{args.algo}_{args.epoch}.json'
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        sys.exit(0)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    ref_model = args.ref_model
    train_model = args.train_model


    data_dir_loc = os.path.join(os.getenv('AMLT_DATA_DIR', "./data/"))

    output_data = []
    d_prompts = []
    dbar_prompts = []

    dataset = load_from_disk(f"./datasets/llama3.1_8B_ultrafeedback/{args.algo}_dataset_1")
    
    
    for item in dataset["train"]:
        if item["type"] == "D":
            d_prompts.append(item["prompt"])
        else:
            dbar_prompts.append(item["prompt"])
                
                
    d_prompts = sorted(list(set(d_prompts)))
    dbar_prompts = sorted(list(set(dbar_prompts)))

    prompt_resp_dict = {}
    
    # ref_llm = LLM(model=ref_model)
    # ref_tokenizer = ref_llm.get_tokenizer()

    # conversations = [ref_tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in d_prompts]
    # sampling_params = SamplingParams(temperature=args.temperature, 
    #                                 top_p=args.top_p, 
    #                                 max_tokens=args.max_tokens, 
    #                                 seed=args.seed,)
    # outputs = ref_llm.generate(conversations, sampling_params)

    # for i, output in enumerate(outputs):
    #     prompt_resp_dict[d_prompts[i]] = output.outputs[0].text

    # del ref_llm
    # del ref_tokenizer
    

    # train_llm = LLM(model=train_model)
    # train_tokenizer = train_llm.get_tokenizer()

    # conversations = [train_tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in dbar_prompts]
    # sampling_params = SamplingParams(temperature=args.temperature, 
    #                                 top_p=args.top_p, 
    #                                 max_tokens=args.max_tokens, 
    #                                 seed=args.seed,)
    # outputs = train_llm.generate(conversations, sampling_params)

    # for i, output in enumerate(outputs):
    #     prompt_resp_dict[dbar_prompts[i]] = output.outputs[0].text
    
    # del train_llm
    # del train_tokenizer

        
    with jsonlines.open(f"{data_dir_loc}/all_train_data.json") as reader:
        for obj in reader:
            if obj["prompt"] in prompt_resp_dict:
                obj["rejected"][1]["content"] = prompt_resp_dict[obj["prompt"]]
            output_data.append(obj)


    # output_file = f'all_train_data_{args.epoch}.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    with jsonlines.open(os.path.join(args.output_dir, output_file), 'w') as writter:
        writter.write_all(output_data)

    print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
    

    def map_to_new_resp(item):
        if item["prompt"] in prompt_resp_dict:
            item["rejected"][1]["content"] = prompt_resp_dict[item["prompt"]]
        return item

    dataset["train"] = dataset["train"].map(lambda item: map_to_new_resp(item))
    dataset.save_to_disk(os.path.join(args.output_dir, f"{args.algo}_dataset_{args.epoch}"))
    sys.exit(0)