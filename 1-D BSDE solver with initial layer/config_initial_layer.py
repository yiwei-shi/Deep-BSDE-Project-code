import numpy as np


class Config(object):
    batch_size = 256
    valid_size = 25600
    num_iterations = 10000
    logging_frequency = 1000          # print result every 1000 training steps
    y_init_range = [0, 1]

class EuropeanCallConfig(Config):
    num_iterations = 20000
    dim = 1
    total_time = 1
    num_time_interval = 10
    lr_values = list(np.array([1e-3, 1e-3]))
    lr_boundaries = [1000]
    pre_train_num_iteration = 5000
    f_layernum = 20
    z_layernum = 20
    f_units = [dim + 20]*f_layernum
    z_units = [dim + 20]*z_layernum


def get_config(name):
    try:
        return globals()[name+'Config']
    except KeyError:
        raise KeyError("Config for the required problem not found.")