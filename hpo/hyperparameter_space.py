d_model_space = [512, 256, 128, 32]
d_ff_space = [2048, 1024, 512, 128]
learning_rate_space = [1e-4, 1e-3]
batch_size_space = [128, 32, 16]
e_layers_space = [1, 2, 3]
d_layers_space = [1, 2, 3]
n_heads_space = [4, 8, 16]
factor_space = [1, 2, 3]

model_space_dict = {
    'iTransformer': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, factor_space],
    'Autoformer': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, factor_space, d_layers_space],
    'Crossformer': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, factor_space, n_heads_space],
    'PatchTST': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, factor_space, n_heads_space],
    'SCINet': [d_model_space,learning_rate_space, batch_size_space],
    'TiDE': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, d_layers_space],
    'TimeXer': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space, factor_space],
    'TimesNet': [d_model_space, d_ff_space, learning_rate_space, batch_size_space, e_layers_space],
}
