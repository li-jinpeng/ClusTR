分两步：
第一步生成数据，更改generate_data.py里的预测长度、数据文件路径、数据名称，运行
第二步运行run.py，增加了--aug是否数据增广、--early_stop是否早停、--cluster_amonut聚类数量、--cluster_index加载第几个聚类的数据集做实验（取值范围0～cluster_amonut-1， 当设置cluster_index==cluster_amonut时，加载全量数据集的训练集和验证集训练，这时需指定test_index，用第几个聚类的测试集做测试）
