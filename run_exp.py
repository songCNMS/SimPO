import os
from scripts.run_simpo import main

epoch = 5

metrics_list = []

train_model = "meta-llama/Llama-3.2-3B-Instruct"
ref_model = "meta-llama/Llama-3.2-3B-Instruct"
os.system(f"python decode_data.py --train_model {train_model} --ref_model {ref_model} --epoch 1")

for ep in range(1, epoch+1):
    print(f"EPOCH: {ep}")
    metrics, output_dir = main(ep)
    metrics_list.append(metrics)
    with open("metrics.jsonl", "w") as writter:
        writter.write_all(metrics_list)
    train_model = output_dir
    if ep <= epoch:
        os.system(f"python decode_data.py --train_model {train_model} --ref_model {ref_model} --epoch {ep}")
