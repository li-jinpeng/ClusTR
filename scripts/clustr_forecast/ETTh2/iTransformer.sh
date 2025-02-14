export CUDA_VISIBLE_DEVICES=$1

model_name=iTransformer
cluster_amount=$2
cluster_domain=time
# cluster_domain=frequency

for pred_len in 96 192 336 720
# for pred_len in 96
    do
        for ((cluster_index=0; cluster_index<$cluster_amount; cluster_index++))
            do
                python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh2.csv \
                --model_id ETTh2_96_$pred_len \
                --model $model_name \
                --data clustr \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len $pred_len \
                --e_layers 2 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 1 \
                --dec_in 1 \
                --c_out 1 \
                --des 'Exp' \
                --d_model 128 \
                --d_ff 128 \
                --itr 1 \
                --cluster_index $cluster_index \
                --cluster_amount $cluster_amount \
                --cluster_domain $cluster_domain
        done
done
