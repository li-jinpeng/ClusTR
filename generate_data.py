from sklearn.preprocessing import StandardScaler
import pandas as pd
from utils.timefeatures import time_features
from tslearn.clustering import TimeSeriesKMeans
import numpy as np

scaler = StandardScaler()
timeenc =  1

data_name = 'ETTh1'
cluster_amount = 2
pred_len = 720
df_raw = pd.read_csv('/data/lijinpeng/ClusTR-v3/dataset/ETT-small/ETTh1.csv')
if 'ETTh' in data_name:
    border1s = [0, 12 * 30 * 24 - 96, 12 * 30 * 24 + 4 * 30 * 24 - 96]
    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
elif 'ETTm' in data_name:
    border1s = [0, 12 * 30 * 24 * 4 - 96, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - 96]
    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
else:
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - 96, len(df_raw) - num_test - 96]
    border2s = [num_train, num_train + num_vali, len(df_raw)]

cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
    
train_data = df_data[border1s[0]:border2s[0]]
scaler.fit(train_data.values)
data = scaler.transform(df_data.values)
train_data = data[border1s[0]:border2s[0],:]
val_data = data[border1s[1]:border2s[1],:]
test_data = data[border1s[2]:border2s[2],:]

num_features = data.shape[1]

df_stamp = df_raw[['date']][border1s[0]:border2s[0]]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
if timeenc == 0:
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], 1).values
elif timeenc == 1:
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
    data_stamp = data_stamp.transpose(1, 0)

num_samples = train_data.shape[0] - 96 - pred_len + 1
data_x = []
for sample in range(num_samples):
    for feature in range(num_features):
        data_x.append(train_data[sample:sample+96, feature])
km = TimeSeriesKMeans(n_clusters=cluster_amount, verbose=True, n_jobs=128, random_state=2021, tol=1e-6).fit(data_x)
centers = km.cluster_centers_.reshape(cluster_amount, -1)

x = [[] for _ in range(cluster_amount+1)]
y = [[] for _ in range(cluster_amount+1)]
x_mark = [[] for _ in range(cluster_amount+1)]
y_mark = [[] for _ in range(cluster_amount+1)]
for sample in range(num_samples):
    for feature in range(num_features):
        seq_x = train_data[sample:sample+96, feature:feature+1]
        seq_y = train_data[sample+96-48:sample+96+pred_len, feature:feature+1]
        seq_x_mark = data_stamp[sample:sample+96]
        seq_y_mark = data_stamp[sample+96-48:sample+96+pred_len]
        
        x[-1].append(seq_x)
        y[-1].append(seq_y)
        x_mark[-1].append(seq_x_mark)
        y_mark[-1].append(seq_y_mark)
        
        distances = np.linalg.norm(centers - seq_x.reshape(-1), axis=1)
        closest_cluster_index = np.argmin(distances)
        x[closest_cluster_index].append(seq_x)
        y[closest_cluster_index].append(seq_y)
        x_mark[closest_cluster_index].append(seq_x_mark)
        y_mark[closest_cluster_index].append(seq_y_mark)

import os
try:
    os.mkdir(data_name)
except:
    pass
try:
    os.mkdir(os.path.join(data_name, f'{cluster_amount}_{pred_len}'))
except:
    pass
dir_path = os.path.join(data_name, f'{cluster_amount}_{pred_len}')

print('train')
for i in range(cluster_amount+1):
    print(len(x[i]))
    np.save(os.path.join(dir_path, f'{i}_train_x'), np.array(x[i]))
    np.save(os.path.join(dir_path, f'{i}_train_y'), np.array(y[i]))
    np.save(os.path.join(dir_path, f'{i}_train_x_mark'), np.array(x_mark[i]))
    np.save(os.path.join(dir_path, f'{i}_train_y_mark'), np.array(y_mark[i]))

x = [[] for _ in range(cluster_amount+1)]
y = [[] for _ in range(cluster_amount+1)]
x_mark = [[] for _ in range(cluster_amount+1)]
y_mark = [[] for _ in range(cluster_amount+1)]
df_stamp = df_raw[['date']][border1s[1]:border2s[1]]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
if timeenc == 0:
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], 1).values
elif timeenc == 1:
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
    data_stamp = data_stamp.transpose(1, 0)
    
num_samples = val_data.shape[0] - 96 - pred_len + 1

for sample in range(num_samples):
    for feature in range(num_features):
        seq_x = val_data[sample:sample+96, feature:feature+1]
        seq_y = val_data[sample+96-48:sample+96+pred_len, feature:feature+1]
        seq_x_mark = data_stamp[sample:sample+96]
        seq_y_mark = data_stamp[sample+96-48:sample+96+pred_len]
        
        x[-1].append(seq_x)
        y[-1].append(seq_y)
        x_mark[-1].append(seq_x_mark)
        y_mark[-1].append(seq_y_mark)
        
        distances = np.linalg.norm(centers - seq_x.reshape(-1), axis=1)
        closest_cluster_index = np.argmin(distances)
        x[closest_cluster_index].append(seq_x)
        y[closest_cluster_index].append(seq_y)
        x_mark[closest_cluster_index].append(seq_x_mark)
        y_mark[closest_cluster_index].append(seq_y_mark)

print('val')   
for i in range(cluster_amount+1):
    print(len(x[i]))
    np.save(os.path.join(dir_path, f'{i}_val_x'), np.array(x[i]))
    np.save(os.path.join(dir_path, f'{i}_val_y'), np.array(y[i]))
    np.save(os.path.join(dir_path, f'{i}_val_x_mark'), np.array(x_mark[i]))
    np.save(os.path.join(dir_path, f'{i}_val_y_mark'), np.array(y_mark[i]))


x = [[] for _ in range(cluster_amount)]
y = [[] for _ in range(cluster_amount)]
x_mark = [[] for _ in range(cluster_amount)]
y_mark = [[] for _ in range(cluster_amount)]     
df_stamp = df_raw[['date']][border1s[2]:border2s[2]]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
if timeenc == 0:
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(['date'], 1).values
elif timeenc == 1:
    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
    data_stamp = data_stamp.transpose(1, 0)
    
num_samples = test_data.shape[0] - 96 - pred_len + 1

for sample in range(num_samples):
    for feature in range(num_features):
        seq_x = test_data[sample:sample+96, feature:feature+1]
        seq_y = test_data[sample+96-48:sample+96+pred_len, feature:feature+1]
        seq_x_mark = data_stamp[sample:sample+96]
        seq_y_mark = data_stamp[sample+96-48:sample+96+pred_len]
        
        distances = np.linalg.norm(centers - seq_x.reshape(-1), axis=1)
        closest_cluster_index = np.argmin(distances)
        x[closest_cluster_index].append(seq_x)
        y[closest_cluster_index].append(seq_y)
        x_mark[closest_cluster_index].append(seq_x_mark)
        y_mark[closest_cluster_index].append(seq_y_mark)
    
print('test')   
for i in range(cluster_amount):
    print(len(x[i]))
    np.save(os.path.join(dir_path, f'{i}_test_x'), np.array(x[i]))
    np.save(os.path.join(dir_path, f'{i}_test_y'), np.array(y[i]))
    np.save(os.path.join(dir_path, f'{i}_test_x_mark'), np.array(x_mark[i]))
    np.save(os.path.join(dir_path, f'{i}_test_y_mark'), np.array(y_mark[i]))