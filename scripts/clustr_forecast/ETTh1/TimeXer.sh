export CUDA_VISIBLE_DEVICES=$1
cluster_amount=$2
cluster_domain=time
model_name=TimeXer

for ((cluster_index=0; cluster_index<=$cluster_amount; cluster_index++))
    do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96 \
      --model $model_name \
      --data clustr \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model 256 \
      --des 'exp' \
      --itr 1 --inverse --cluster_index $cluster_index \
                    --cluster_amount $cluster_amount \
                    --cluster_domain $cluster_domain \



    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_192 \
      --model $model_name \
      --data clustr \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 192 \
      --e_layers 2 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --d_model 128 \
      --itr 1 --inverse --cluster_index $cluster_index \
                    --cluster_amount $cluster_amount \
                    --cluster_domain $cluster_domain

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_336 \
      --model $model_name \
      --data clustr \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 336 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 1024 \
      --itr 1 --inverse --cluster_index $cluster_index \
                    --cluster_amount $cluster_amount \
                    --cluster_domain $cluster_domain

    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_720 \
      --model $model_name \
      --data clustr \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 720 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --d_model 256 \
      --d_ff 1024 \
      --itr 1 --inverse --cluster_index $cluster_index \
                    --cluster_amount $cluster_amount \
                    --cluster_domain $cluster_domain
  done