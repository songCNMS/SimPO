

for algo in simpo alphaDPO alphaDPO-DA
do
    conda run -n simpo python scripts/init_decode_data.py --train_model meta-llama/Llama-3.2-3B-Instruct --ref_model meta-llama/Llama-3.2-3B-Instruct --epoch 1 --algo $algo;

    for i in {1..4}; do \
        conda run -n simpoenv python scripts/run_simpo.py training_configs/llama-3-3b-instruct-$algo-v2.yaml epoch=$i;\
        conda run -n simpoenv python scripts/run_simpo_eval.py training_configs/llama-3-3b-instruct-$algo-v2.yaml  epoch=$i exp_name=$algo-$i;\
        conda run -n simpo python scripts/decode_data.py --ref_model meta-llama/Llama-3.2-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/$algo_$i --epoch $((i+1))  --algo $algo;\
    done
done