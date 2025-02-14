import os
gpu_index = 0
cluster_amount=3
data_paths = {
    'ETTh1': 'ETT-small',
    'ETTh2': 'ETT-small',
    'exchange_rate': 'exchange_rate',
}
import shutil

regenerate_data = 1

# for data in ['ETTh2']:
#     for model in ['iTransformer']:
#         gpu_index %= 8
#         if not os.path.exists(f'./logs/{data}'):
#             os.mkdir(f'./logs/{data}')
#         script = f'nohup bash ./scripts/clustr_baseline/{data}/{model}.sh {gpu_index} > ./logs/{data}/{model}_baseline.log 2>&1 &'
#         os.system(script)
#         gpu_index += 1


# for data in ['ETTh1', 'ETTh2']:
#     if regenerate_data:
#         data_path = f'./dataset/{data_paths[data]}/{data}_time_{cluster_amount}'
#         try:
#             shutil.rmtree(data_path)
#         except:
#             pass
#     for model in ['PatchTST', 'iTransformer', 'DLinear']:
#         gpu_index %= 8
#         if not os.path.exists(f'./logs/{data}'):
#             os.mkdir(f'./logs/{data}')
#         script = f'nohup bash ./scripts/clustr_forecast/{data}/{model}.sh {gpu_index} {cluster_amount} > ./logs/{data}/{model}_{cluster_amount}.log 2>&1 &'
#         os.system(script)
#         gpu_index += 1

for data in ['ETTh1']:
    if regenerate_data:
        data_path = f'./dataset/{data_paths[data]}/{data}_time_{cluster_amount}'
        try:
            shutil.rmtree(data_path)
        except:
            pass
    for model in ['iTransformer']:
        gpu_index %= 8
        if not os.path.exists(f'./logs/{data}'):
            os.mkdir(f'./logs/{data}')
        script = f'nohup bash ./scripts/clustr_forecast/{data}/{model}.sh {gpu_index} {cluster_amount} > ./logs/{data}/{model}_{cluster_amount}.log 2>&1 &'
        os.system(script)
        gpu_index += 1