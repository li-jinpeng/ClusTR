export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_96 \
#   --model $model_name \
#   --data clustr \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --des 'Exp' \
#   --d_model 128 \
#   --d_ff 128 \
#   --itr 1 \
#   --cluster_amount 2 \
#   --cluster_index 0 \
#   --test_index 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data clustr \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 32 \
  --batch_size 16 \
  --d_ff 64 \
  --itr 1 \
  --cluster_amount 3 \
  --cluster_index 0 \
  --test_index 0 \
