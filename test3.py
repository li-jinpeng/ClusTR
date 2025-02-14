

import numpy as np
from skfuzzy.cluster import cmeans
from skfuzzy.cluster import cmeans_predict

# 生成示例数据（3个簇，2维特征）

import tslearn.metrics as metrics
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix
import pandas as pd


df_raw = pd.read_csv('/data/lijinpeng/ClusTR/dataset/ETT-small/ETTh1.csv')
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
data = df_data.values

threshold = 0.9
num_samples = data.shape[0] - 96 - 96 + 1
num_features = data.shape[1]

X = []
for sample in range(num_samples):
    for feature in range(num_features):
        seq_x = data[sample:sample+96, feature]
        X.append(seq_x)
X = np.array(X)
X_TEST = X[-len(X)//5:]


# 设置参数
n_clusters = 3
m = 2.0  # 模糊因子（越大越模糊）

# 运行FCM
cntr, u, _, _, _, _, _ = cmeans(
    X.T, n_clusters, m, error=0.001, maxiter=1000, seed=2021
)

# 获取样本的隶属度矩阵（每行对应一个样本，每列对应簇的隶属度）
membership = u.T

# 判断样本所属簇（隶属度 > 阈值则视为属于该簇）
overlapping_clusters = [
    np.where(membership[i] > threshold)[0] for i in range(len(X))
]

a = [0,0,0]

for o in overlapping_clusters:
    for m in o:
        a[m] += 1

print(len(X))
print(a)

t = [0,0,0]
for o in membership:
    for i in range(3):
        if o[i] < 0.1:
            t[i] += 1
    
print(t)

membership = np.array(membership)
print(np.mean(membership, axis=0))
        

print("样本0的隶属度:", membership[len(X)//2])
print("样本0所属的簇:", overlapping_clusters[len(X)//2])

k = 0
u_new, _, _, _, _, _ = cmeans_predict(
    X_TEST.T, cntr, m, error=0.001, maxiter=1000, seed=2021
)


membership = u_new.T
overlapping_clusters = [
    np.where(membership[i] > threshold)[0] for i in range(len(X_TEST))
]

threshold = 0.6
a = [0,0,0]
b =[0,0,0]
kk = 0
for m in membership:
    a[np.argmax(m)] += 1
    if m[np.argmax(m)] < threshold:
        b[np.argmax(m)] += 1
    flag = False
    for o in m:
        if o > threshold:
            flag = True
            break
    if not flag:
        kk += 1

print(len(X_TEST))
print(a)
print(b)
print(kk)
    
