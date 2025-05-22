import torch
import random
import numpy as np
import yaml

def get_config(dataset_name):
    with open('./configs/%s.yaml'%(dataset_name), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False