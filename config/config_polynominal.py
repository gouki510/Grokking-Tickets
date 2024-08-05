import os
import numpy as np
from pathlib import Path

class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-2
        self.weight_decay = 1
        self.p = 67
        self.d_emb = 500
        self.d_model = 48 #128#48#48
        self.frac_train = 0.4
        self.is_symmetric_input = True #False if 'subtract' in self.fn_name else True
        self.num_epochs = 10000
        self.save_models = True 
        self.save_every = 200

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 0
        self.root = Path("Neurips_rebuttal_polynominal")
        self.model = 'mlp' # ['mlp', 'transformer']
        self.is_1layer = False
        os.makedirs(self.root,exist_ok=True)

        self.num_layers =  0
        self.batch_size = 100
        self.d_vocab = 40
        self.d_mlp = 1*self.d_model
        self.act_type = 'ReLU'

        self.use_ln = False
        self.optimizer = 'adam' # ['sgd', 'adam']
        self.weight_scale = 1

        
        self.exp_name = f"{self.optimizer}_WD_{self.weight_decay}_LR_{self.lr}"