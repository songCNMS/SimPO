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


parser = argparse.ArgumentParser(description="Decode with vllm")
parser.add_argument(
    "--train_model",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="Path to the LLM model",
)
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
parser.add_argument("--algo", type=str, default="alphaDPO", help="Algo.")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--ori_rej", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

    print(args)

    output_file = f"all_train_data_{args.algo}_{args.epoch}.json"
    if os.path.exists(os.path.join(args.output_dir, output_file)):
        sys.exit(0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ref_model = args.ref_model
    train_model = args.train_model

    data_dir_loc = os.path.join(os.getenv("AMLT_DATA_DIR", "./data/"))

    d_prompts = []
    dbar_prompts = []
    with jsonlines.open(f"{data_dir_loc}/all_train_data.json") as reader:
        for obj in reader:
            if obj["type"] == "D":
                d_prompts.append(obj["prompt"])
            else:
                dbar_prompts.append(obj["prompt"])

    d_prompts = sorted(list(set(d_prompts)))
    dbar_prompts = sorted(list(set(dbar_prompts)))
    if len(dbar_prompts) > len(d_prompts)*0.2:
        dbar_prompts = np.random.choice(
            dbar_prompts, size=int(len(d_prompts)*0.2), replace=False
        )

    all_prompts = list(d_prompts) + list(dbar_prompts)
    if args.debug and args.num_samples < len(all_prompts):
        all_prompts = np.random.choice(all_prompts, size=args.num_samples, replace=False)
    
    prompt_resp_dict = defaultdict(list)

    ref_llm = LLM(model=ref_model)
    ref_tokenizer = ref_llm.get_tokenizer()
    conversations = [
        ref_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in all_prompts
    ]
    
    
    
    for i in range(3):
        sampling_params = SamplingParams(
            # temperature=args.temperature,
            temperature=i/2.0,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            # logprobs=20,
            seed=args.seed + i,
        )
        
        outputs = ref_llm.generate(conversations, sampling_params)

        for i, output in enumerate(outputs):
            prompt_resp_dict[all_prompts[i]].append((1.0, output.outputs[0].text))
            # prompt_resp_dict[all_prompts[i]].append((output.outputs[0].cumulative_logprob, output.outputs[0].text))
            # if i < len(d_prompts):
            #     prompt_resp_dict[d_prompts[i]].append(output.outputs[0].text)
            # else:
            #     prompt_resp_dict[dbar_prompts[i-len(d_prompts)]].append(output.outputs[0].text)

    del ref_llm
    del ref_tokenizer

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

    prompt_set = set()
    
    output_train_data = []
    output_test_data = []
    
    with jsonlines.open(f"{data_dir_loc}/all_train_data.json") as reader:
        for obj in reader:
            if obj["prompt"] in prompt_set: continue
            if obj["prompt"] in prompt_resp_dict:
                # candidate_prompts = sorted(prompt_resp_dict[obj["prompt"]], key=lambda x: x[0], reverse=True)
                candidate_prompts = prompt_resp_dict[obj["prompt"]]
                if not args.ori_rej:
                    obj["rejected"][1]["content"] = random.choice(candidate_prompts)[1]
                if obj["type"] == "DBAR":
                    obj["alpha"] = 1.0
                    if random.random() >= 0.2:
                        obj["rejected"][1]["content"] = candidate_prompts[-1][1]
                        obj["chosen"][1]["content"] = candidate_prompts[0][1]
                        output_train_data.append(obj)
                    else:
                        candidate_prompts = list(set([x[1] for x in candidate_prompts]))
                        if len(candidate_prompts) <= 1:
                            candidate_prompts.extend([obj["rejected"][1]["content"], obj["chosen"][1]["content"]])
                        obj["rejected"][1]["content"] = random.choice(candidate_prompts)
                        obj["chosen"][1]["content"] = random.choice(candidate_prompts)
                        output_test_data.append(obj)
                else:
                    obj["alpha"] = 0.0
                    if random.random() <= 0.2:
                        output_test_data.append(obj)
                    else:
                        output_train_data.append(obj)
                prompt_set.add(obj["prompt"])

    # with open(os.path.join(args.output_dir, output_file), 'w') as f:
    #     json.dump(output_data, f, indent=4)

    with jsonlines.open(os.path.join(args.output_dir, output_file), "w") as writter:
        writter.write_all(output_train_data + output_test_data)

    print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
    
    dataset = datasets.DatasetDict({
        "train": datasets.Dataset.from_list(output_train_data),
        "test": datasets.Dataset.from_list(output_test_data)
    })
    # if args.debug and args.num_samples < len(dataset):
    #     dataset = dataset.shuffle(seed=42).select(range(args.num_samples))
    # dataset = dataset.train_test_split(test_size=0.2)
    dataset.save_to_disk(
        os.path.join(args.output_dir, f"{args.algo}_dataset_{args.epoch}")
    )
    sys.exit(0)
