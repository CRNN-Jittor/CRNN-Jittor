import os

curr_path = os.path.dirname(__file__)

common_config = {
    'data_dir': os.path.join(curr_path, '../data/Synth90k/'),
    'test_dir': os.path.join(curr_path, '../data/'),
    'img_width': 100,
    'img_height': 32,
    'rnn_hidden': 256,
    'leaky_relu': False,
    'cpu_workers': 16,
}

train_config = {
    'epochs': 10000,
    'train_batch_size': 32,
    'eval_batch_size': 512,
    'lr': 0.0005,
    'show_interval': 10,
    'valid_interval': 500,
    'save_interval': 2000,
    'reload_checkpoint': None,
    'valid_max_iter': 100,
    'decode_method': 'greedy',
    'beam_size': 10,
    'checkpoints_dir': os.path.join(curr_path, '../checkpoints/')
}
train_config.update(common_config)

evaluate_config = {
    'eval_batch_size': 512,
    'reload_checkpoint': os.path.join(curr_path, '../checkpoints/crnn_synth90k.pt'),
    'decode_method': 'beam_search',
    'beam_size': 10,
}
evaluate_config.update(common_config)
