from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import os
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
import on_policy_data_gen.gpt_api_config
import torch

# mistralai/Mistral-7B-Instruct-v0.2
# google/gemma-2-9b-it
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen2.5-7B-Instruct

all_ref_model_names = ["llama3-3b", "qwen25-3b", "mistral-7b", "gemma2-9b"]
all_ref_models = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "google/gemma-2-9b-it"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Decode with vllm')
    parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                        help='Directory containing the data')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help='Path to the LLM model')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Temperature for sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p probability for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default="llama3.1_8B_ultrafeedback",
                        help='output_dir')
    args = parser.parse_args()




    ref_model_name = all_ref_model_names[all_ref_models.index(args.model)]

    # os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"), "./")
    # os.path.join(os.getenv('AMLT_DATA_DIR', "./"), "./")
    args.output_dir = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"), f"./data/{ref_model_name}/")

    data_dir = args.data_dir
    llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count())
    tokenizer = llm.get_tokenizer()

    train_dataset= load_dataset(data_dir, split='train_prefs')

    prompts = sorted(list(set(train_dataset['prompt'])))

    conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

    sampling_params = SamplingParams(temperature=args.temperature, 
                                    top_p=args.top_p, 
                                    max_tokens=args.max_tokens, 
                                    seed=args.seed,)
    outputs = llm.generate(conversations, sampling_params)

    # Save the outputs as a JSON file.
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        output_data.append({
            'prompt': prompts[i],
            "format_prompt": prompt,
            'generated_text': generated_text,
        })

    output_file = f'output_{args.seed}.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")
