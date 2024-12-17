import json
import argparse
import os
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Diretory containing the generation files",
    default="llama",
)
args = parser.parse_args()

print(args)

generation_file_dir = f"datasets/{args.model}"


all_data = []
for file_name in glob(
    f"./amlt/dpo_data_ultrafeedback/*/data/{args.model}/output_*.json"
):
    generation_file = file_name
    print(file_name)
    with open(generation_file, "r") as f:
        output_data = json.load(f)
        all_data.append(output_data)

num_samples = len(all_data[0])
all_res = []
num_identical = 0
for i in range(num_samples):
    prompt = all_data[0][i]["prompt"]
    gen_text = []
    for data in all_data:
        gen_text.append(data[i]["generated_text"])

    if len(set(gen_text)) == 1:
        # filter out samples where all generated responses are identical
        num_identical += 1
        continue

    all_res.append(
        {
            "prompt": prompt,
            "all_generated_responses": gen_text,
        }
    )

print(f"Filtered out {num_identical} samples with identical generated responses")

os.makedirs(generation_file_dir, exist_ok=True)
with open(os.path.join(generation_file_dir, "all_outputs.json"), "w") as f:
    json.dump(all_res, f, indent=4)

print(
    f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_outputs.json')}"
)
