import json
import argparse
import os
from glob import glob
import jsonlines
from json import encoder

encoder.FLOAT_REPR = lambda o: format(o, '.14f')

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
    f"/home/lesong/codes/SimPO/amlt/dpo_mp_data/*/mh/all_train_data_{args.model}_*.json"
):
    generation_file = file_name
    print(file_name)
    with jsonlines.open(generation_file, "r") as reader:
        # output_data = json.load(f)
        all_data.append([obj for obj in reader])

num_samples = len(all_data[0])
all_res = []
num_identical = 0
for i in range(num_samples):
    prompt = all_data[0][i]["prompt"]
    gen_text = []
    scores = []
    for data in all_data:
        gen_text.append(data[i]["model_response"])
        scores.append(data[i]["model_response_logprob"])

    if len(set(gen_text)) == 1:
        # filter out samples where all generated responses are identical
        num_identical += 1
        continue

    all_res.append(
        {
            "prompt": prompt,
            "type": all_data[0][i]["type"],
            "all_generated_responses": gen_text,
            "all_rm_scores": scores,
        }
    )

print(f"Filtered out {num_identical} samples with identical generated responses")

os.makedirs(generation_file_dir, exist_ok=True)
with open(os.path.join(generation_file_dir, "all_outputs.json"), "w") as f:
    json.dump(all_res, f, indent=4)

print(
    f"Processed outputs saved to {os.path.join(generation_file_dir, 'all_outputs.json')}"
)
