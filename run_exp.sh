



conda run -n simpo_data python scripts/init_decode_data.py --train_model meta-llama/Llama-3.2-3B-Instruct --ref_model meta-llama/Llama-3.2-3B-Instruct --epoch 1 --algo cpo --num_samples 10000 --debug --output_dir datasets/llama3_3b_ultrafeedback --ori_rej;

for i in {1..1}
do
    for algo in DPO SimPO alphaDPO
    do
        conda run -n simpo_data python scripts/decode_data.py --ref_model meta-llama/Llama-3.2-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/$algo_$i --epoch $i  --algo $algo --output_dir datasets/llama3_3b_ultrafeedback;\
        conda run -n simpo python scripts/run_simpo.py training_configs/llama-3-3b-instruct-$algo-v2.yaml epoch=$i;\
        conda run -n simpo python scripts/run_simpo_eval.py training_configs/llama-3-3b-instruct-$algo-v2.yaml  epoch=$i exp_name=$algo-$i run_name=llama3b_original_rejected_more_data;\
    done
done


conda run -n simpo_data python scripts/init_decode_data.py --train_model Qwen/Qwen2.5-3B-Instruct --ref_model Qwen/Qwen2.5-3B-Instruct --epoch 1 --algo cpo --num_samples 10000 --debug --output_dir datasets/qwen25_3b_ultrafeedback --ori_rej;

for i in {1..1}
do
    for algo in DPO SimPO alphaDPO
    do
        conda run -n simpo_data python scripts/decode_data.py --ref_model Qwen/Qwen2.5-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/$algo_$i --epoch $i  --algo $algo --output_dir datasets/qwen25_3b_ultrafeedback;\
        conda run -n simpo python scripts/run_simpo.py training_configs/qwen25-3b-instruct-$algo-v2.yaml epoch=$i;\
        conda run -n simpo python scripts/run_simpo_eval.py training_configs/qwen25-3b-instruct-$algo-v2.yaml  epoch=$i exp_name=$algo-$i run_name=qwen25_3b_original_rejected_more_data;\
    done
done



conda run -n simpo_data python scripts/init_decode_data.py --train_model meta-llama/Llama-3.2-3B-Instruct --ref_model meta-llama/Llama-3.2-3B-Instruct --epoch 1 --algo cpo --num_samples 10000 --debug --output_dir datasets/llama3_3b_ultrafeedback;

for i in {1..1}
do
    for algo in SFTReg DPO SimPO alphaDPO
    do
        conda run -n simpo_data python scripts/decode_data.py --ref_model meta-llama/Llama-3.2-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/$algo_$i --epoch $i  --algo $algo --output_dir datasets/llama3_3b_ultrafeedback;\
        conda run -n simpo python scripts/run_simpo.py training_configs/llama-3-3b-instruct-$algo-v2.yaml epoch=$i;\
        conda run -n simpo python scripts/run_simpo_eval.py training_configs/llama-3-3b-instruct-$algo-v2.yaml  epoch=$i exp_name=$algo-$i run_name=llama3b_ref_rejected_more_data;\
    done
done


conda run -n simpo_data python scripts/init_decode_data.py --train_model Qwen/Qwen2.5-3B-Instruct --ref_model Qwen/Qwen2.5-3B-Instruct --epoch 1 --algo cpo --num_samples 10000 --debug --output_dir datasets/qwen25_3b_ultrafeedback;

for i in {1..1}
do
    for algo in SFTReg DPO SimPO alphaDPO
    do
        conda run -n simpo_data python scripts/decode_data.py --ref_model Qwen/Qwen2.5-3B-Instruct --train_model /home/lesong/codes/SimPO/outputs/$algo_$i --epoch $i  --algo $algo --output_dir datasets/qwen25_3b_ultrafeedback;\
        conda run -n simpo python scripts/run_simpo.py training_configs/qwen25-3b-instruct-$algo-v2.yaml epoch=$i;\
        conda run -n simpo python scripts/run_simpo_eval.py training_configs/qwen25-3b-instruct-$algo-v2.yaml  epoch=$i exp_name=$algo-$i run_name=qwen25_3b_ref_rejected_more_data;\
    done
done