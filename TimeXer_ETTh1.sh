export CUDA_VISIBLE_DEVICES=0
model_name=TimeXer
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
  --e_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 72 \
  --batch_size 16 \
  --itr 1 \
  --cluster_amount 2 \
  --cluster_index 1 \
  --test_index 0 \
  --early_stop 0 \
  --train_epochs 2

# 0.525 
# --d_model 64 \
#   --d_ff 512 \
#   --batch_size 16 \
# d_model 72