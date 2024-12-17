from string import Template
import os
from omegaconf import OmegaConf
from on_policy_data_gen.decode import all_ref_model_names, all_ref_models


config_temp = Template(
    """
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
max_prompt_length: 5120
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
save_total_limit: 1
seed: 42
warmup_ratio: 0.1"""
)

os.makedirs("batch_trainer_configs", exist_ok=True)
all_trainer_types = ["DPO-sigmoid", "alphaDPO", "SimPO", "IPO", "KTO", 'rDPO', 'SFTReg', "SFTRegWoTRef"]
all_loss_types = ["sigmoid", "alpha-dpo", "simpo", "ipo", "kto", "rDPO", "sft-reg", "sft-reg-wot-ref"]




cfg = OmegaConf.from_cli()
task = cfg.get("task", "data")
ref_models = cfg.ref_models.split(",")
ref_model_names = [all_ref_model_names[all_ref_models.index(ref_model)] for ref_model in ref_models]
alpha = cfg.get("alpha", 0.4)
beta = cfg.get("beta", 1.0)

output_dir = os.path.join(os.getenv('AMLT_OUTPUT_DIR', "./"), "./")
data_dir = os.path.join(os.getenv('AMLT_DATA_DIR', "./"), "./")



for ref_model, ref_model_name in zip(ref_models, ref_model_names):
    if task == "data":
        os.system(
            f"python scripts/init_decode_data.py --train_model {ref_model} --ref_model {ref_model} --epoch 1 --algo cpo --num_samples 2000 --debug --output_dir {output_dir}/datasets/{ref_model_name}-ori --ori_rej;"
        )
        os.system(
            f"python scripts/init_decode_data.py --train_model {ref_model} --ref_model {ref_model} --epoch 1 --algo cpo --num_samples 2000 --debug --output_dir {output_dir}/datasets/{ref_model_name};"
        )
    else:
        trainer_types = cfg.trainer_types.split(",")
        loss_types = [all_loss_types[all_trainer_types.index(trainer_type)] for trainer_type in trainer_types]
        for loss_type, trainer_type in zip(loss_types, trainer_types):
            config = config_temp.substitute(
                alpha=alpha,
                beta=beta,
                trainer_type=trainer_type,
                loss_type=loss_type,
                ref_model=ref_model,
                ref_model_name=ref_model_name,
            )
            config_loc = f"batch_trainer_configs/{ref_model_name}-{trainer_type}-beta{beta}-alpha{alpha}.yaml"
            with open(config_loc, "w") as f:
                f.writelines(config)
            for epoch in range(1, 2):
                os.system(
                    f"python scripts/decode_data.py --ref_model {ref_model} --train_model {output_dir}/outputs/{loss_type}-{epoch} --epoch {epoch}  --algo {trainer_type} --output_dir {data_dir}/datasets/{ref_model_name}-ori;"
                )
                os.system(
                    f"python scripts/run_simpo.py {config_loc} epoch={epoch} data_dir={data_dir}/datasets/{ref_model_name}-ori;"
                )
                os.system(
                    f"python scripts/run_simpo_eval.py {config_loc}  epoch={epoch} exp_name={loss_type}-{epoch} data_dir={data_dir}/datasets/{ref_model_name}-ori run_name={ref_model_name}-ori;"
                )

                os.system(
                    f"python scripts/decode_data.py --ref_model {ref_model} --train_model {output_dir}/outputs/{loss_type}-{epoch} --epoch {epoch}  --algo {trainer_type} --output_dir {data_dir}/datasets/{ref_model_name};"
                )
                os.system(
                    f"python scripts/run_simpo.py {config_loc} epoch={epoch} data_dir={data_dir}/datasets/{ref_model_name};"
                )
                os.system(
                    f"python scripts/run_simpo_eval.py {config_loc}  epoch={epoch} exp_name={loss_type}-{epoch} data_dir={data_dir}/datasets/{ref_model_name} run_name={ref_model_name};"
                )