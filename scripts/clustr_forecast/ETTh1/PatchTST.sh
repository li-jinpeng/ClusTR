export CUDA_VISIBLE_DEVICES=$1

model_name=PatchTST
cluster_amount=$2
cluster_domain=time
declare -A dict
dict[96]=2
dict[192]=8
dict[336]=8
dict[720]=16

for pred_len in 96 192 336 720
    do
        for ((cluster_index=0; cluster_index<$cluster_amount; cluster_index++))
            do
                python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path ./dataset/ETT-small/ \
                --data_path ETTh1.csv \
                --model_id ETTh1_96_$pred_len \
                --model $model_name \
                --data clustr \
                --features M \
                --seq_len 96 \
                --label_len 48 \
                --pred_len $pred_len \
                --e_layers 1 \
                --d_layers 1 \
                --factor 3 \
                --enc_in 1 \
                --dec_in 1 \
                --c_out 1 \
                --des 'Exp' \
                --itr 1 \
                --inverse \
                --n_heads ${dict[$pred_len]} \
                --cluster_index $cluster_index \
                --cluster_amount $cluster_amount \
                --cluster_domain $cluster_domain
        done
done
