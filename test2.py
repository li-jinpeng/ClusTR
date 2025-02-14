import networkx as nx
from cdlib import algorithms
import pandas as pd

print(algorithms.__dict__.keys())

import tslearn.metrics as metrics
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix



df_raw = pd.read_csv('/data/lijinpeng/ClusTR/dataset/ETT-small/ETTh1.csv')
cols_data = df_raw.columns[1:]
df_data = df_raw[cols_data]
data = df_data.values

  
num_samples = data.shape[0] - 96 - 96 + 1
num_features = data.shape[1]

X = []
for sample in range(num_samples):
    for feature in range(num_features):
        seq_x = data[sample:sample+96, feature]
        X.append(seq_x)

X = X[:1000]    
num_samples = 1000
# 生成k近邻稀疏邻接矩阵
nbrs = NearestNeighbors(n_neighbors=10, metric="euclidean", algorithm='kd_tree', n_jobs=64).fit(X)
distances, indices = nbrs.kneighbors(X)

import numpy as np
# 构建稀疏矩阵
rows = np.repeat(np.arange(len(X)), 10)
cols = indices.flatten()
data = distances.flatten()

adjacency_sparse = csr_matrix((data, (rows, cols)), shape=(num_samples, num_samples))

# 将相似性矩阵转换为 NetworkX 图     
        
# adjacency_matrix = metrics.cdist_dtw(X, n_jobs=128, verbose=1)
# adjacency_matrix = pairwise_distances(X, metric="euclidean", n_jobs=32)

G = nx.from_numpy_array(adjacency_sparse)

print('BigClam')
# 运行 BigClam 算法
communities = algorithms.big_clam(G, min_community_size=(num_samples*num_features//2))  # 设置最小簇大小

# 获取重叠的簇
overlapping_clusters = communities.communities
print(overlapping_clusters)