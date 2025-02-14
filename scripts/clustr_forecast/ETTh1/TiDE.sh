export CUDA_VISIBLE_DEVICES=$1

model_name=TiDE
cluster_amount=$2
cluster_domain=time
# cluster_domain=frequency

for pred_len in 96 192 336 720
    do
        for ((cluster_index=0; cluster_index<=$cluster_amount; cluster_index++))
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
                  --e_layers 2 \
                  --d_layers 2 \
                  --enc_in 1 \
                  --dec_in 1 \
                  --c_out 1 \
                  --d_model 256 \
                  --d_ff 256 \
                  --dropout 0.3 \
                  --batch_size 512 \
                  --learning_rate 0.1 \
                  --patience 5 \
                  --train_epochs 10 \
                  --inverse \
                  --cluster_index $cluster_index \
                  --cluster_amount $cluster_amount \
                  --cluster_domain $cluster_domain
        done
done

