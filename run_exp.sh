
# conda run -n simpo python scripts/init_decode_data.py --train_model meta-llama/Llama-3.2-3B-Instruct --ref_model meta-llama/Llama-3.2-3B-Instruct --epoch 1;

for i in {1..10}; do \
    conda run -n simpoenv python scripts/run_simpo.py training_configs/llama-3-3b-instruct-simpo-v2.yaml  epoch=$i; \
    conda run -n simpo python scripts/decode_data.py --ref_model meta-llama/Llama-3.2-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/llama-3-3b-instruct-simpo-v2_$i --epoch $((i+1)); \ 
done