import os
import numpy as np
from pathlib import Path

class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-3
        self.weight_decay = 1e-3
        self.num_epochs = 10000
        self.save_models = True 
        self.save_every = 200
        self.lp = 1
        self.lp_alpha = 0
        self.momentum = 0.9
        self.sparse_k = 0.1
        self.optimizer = 'sgd' # ['sgd', 'adam']
        
        # data
        self.img_size = [28,28,1]
        self.frac_train = 0.1
        self.batch_size = 200
        
        # model
        self.model = 'mlp' # ['mlp', 'transformer']
        self.num_layers = 0
        self.d_emb = 200
        self.d_model = 200 #128#48#48
        self.d_vocab = 10 # d_class

        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 9
        self.use_ln = False

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        
        # others
        self.seed = 0
        self.root = Path("0120/slth/mnist") 
        os.makedirs(self.root,exist_ok=True)

        
        # slth
        self.prune_rate = 0.4
        self.weight_learning = True
        self.weight_init_path = None


    
        self.exp_name = f"{self.optimizer}_WD_{self.weight_decay}_LR_{self.lr}_MO_{self.momentum}"