from string import Template
import os

config_temp = Template("""
# Model arguments
model_name_or_path: $ref_model
torch_dtype: null
attn_implementation: sdpa
# attn_implementation: flash_attention_2

# Data training arguments
dataset_mixer:
  /home/lesong/codes/SimPO/datasets/$ref_model_name/: 1.0
dataset_splits:
- train
- test
preprocessing_num_workers: 12

# AlphaDPOTrainer arguments
fp16: true
beta: $beta
gamma_beta_ratio: 0.4
alpha: $alpha
ln: true
trainer_type: $trainer_type
loss_type: $loss_type
do_eval: true
evaluation_strategy: steps
gradient_accumulation_steps: 1
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: False
hub_model_id: $loss_type-exps
learning_rate: 1.0e-6
log_level: info
logging_steps: 1000
lr_scheduler_type: cosine
max_length: 3072
max_prompt_length: 2800
num_train_epochs: 3
optim: adamw_torch
output_dir: outputs/$ref_model_name-alpha-$loss_type-v2
run_name: $ref_model_name-$loss_type-beta$beta-alpha$alpha
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
push_to_hub: false
save_strategy: "steps"
save_steps: 1000000
eval_steps: 1000000
report_to:
- wandb
save_total_limit: 20
seed: 42
warmup_ratio: 0.1""")

os.makedirs("batch_trainer_configs", exist_ok=True)

alpha = 0.4
beta = 1.0
# trainer_types = ["DPO-sigmoid", "alphaDPO", "SimPO", "IPO", "KTO", 'rDPO', 'SFTReg', "SFTRegWoTRef"]
# loss_types = ["sigmoid", "alpha-dpo", "simpo", "ipo", "kto", "rDPO", "sft-reg", "sft-reg-wot-ref"]

trainer_types = ['SFTReg', "SFTRegWoTRef", "IPO", "KTO", 'rDPO']
loss_types = ["sft-reg", "sft-reg-wot-ref", "ipo", "kto", "rDPO"]


ref_model_names = ["llama3-3b", "qwen25-3b"]
ref_models = ["meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]

for loss_type, trainer_type in zip(loss_types, trainer_types):
    for ref_model, ref_model_name in zip(ref_models, ref_model_names):
        cfg = config_temp.substitute(alpha=alpha, beta=beta, trainer_type=trainer_type, loss_type=loss_type, ref_model=ref_model, ref_model_name=ref_model_name)
        config_loc = f"batch_trainer_configs/{ref_model_name}-{trainer_type}-beta{beta}-alpha{alpha}.yaml"
        with open(config_loc, "w") as f:
            f.writelines(cfg)

        for epoch in range(1, 2):
            os.system(f"python scripts/decode_data.py --ref_model {ref_model} --train_model /home/lesong/codes/SimPO/outputs/{loss_type}-{epoch} --epoch {epoch}  --algo {trainer_type} --output_dir datasets/{ref_model_name}-ori;")
            os.system(f"python scripts/run_simpo.py {config_loc} epoch={epoch} data_dir=datasets/{ref_model_name}-ori;")
            os.system(f"python scripts/run_simpo_eval.py {config_loc}  epoch={epoch} exp_name={loss_type}-{epoch} data_dir=datasets/{ref_model_name}-ori {ref_model_name}-{trainer_type}-beta{beta}-alpha{alpha}-ori;")

            os.system(f"python scripts/decode_data.py --ref_model {ref_model} --train_model /home/lesong/codes/SimPO/outputs/{loss_type}-{epoch} --epoch {epoch}  --algo {trainer_type} --output_dir datasets/{ref_model_name};")
            os.system(f"python scripts/run_simpo.py {config_loc} epoch={epoch} data_dir=datasets/{ref_model_name};")
            os.system(f"python scripts/run_simpo_eval.py {config_loc}  epoch={epoch} exp_name={loss_type}-{epoch} data_dir=datasets/{ref_model_name} run_name={ref_model_name}-{trainer_type}-beta{beta}-alpha{alpha};")