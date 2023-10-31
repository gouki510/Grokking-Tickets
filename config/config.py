import os
import numpy as np
from pathlib import Path

class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-3 
        self.weight_decay = 0
        self.p = 67 
        self.d_emb = 500
        self.d_model = 48 
        fn_name = 'add'  #['add', 'subtract', 'x2xyy2','rand']'
        self.is_div = True if "only" in fn_name  else False
        self.frac_train = 0.4
        self.is_symmetric_input = True
        self.num_epochs = 10000
        self.save_models = True 
        self.save_every = 2000

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 2
        self.root = Path("1031/lp_reg") 
        self.model = 'mlp' # ['mlp', 'transformer']
        os.makedirs(self.root,exist_ok=True)

        self.num_layers =  0
        self.batch_style = 'full' # ['full', 'random']
        self.d_vocab = self.p
        self.n_ctx = 2
        self.d_mlp = 1*self.d_model
        self.num_heads = 1
        assert self.d_model % self.num_heads == 0
        self.d_head = self.d_model//self.num_heads
        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 1
        self.prune_rate = 0.4
        self.lp = 1
        self.lp_alpha = 0.003

        self.use_ln = False

        self.random_answers = np.random.randint(low=0, high=self.p, size=(self.p, self.p))

        self.fns_dict = {'add': lambda x,y:(x+y)%self.p, 'subtract': lambda x,y:(x-y)%self.p, \
                         'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p, 'rand':lambda x,y:self.random_answers[x][y],\
                        "only_add": lambda x,y:(x+y)}
        
        self.fn = self.fns_dict[fn_name]
        self.exp_name = f"{self.model}_{self.num_layers}L_{self.d_model}D_{self.p}P_\
            {self.frac_train}F_{self.lr}LR_{self.weight_decay}WD_{self.is_symmetric_input}S_\
            {self.weight_scale}WS_{fn_name}task_{self.lp}LP_{self.lp_alpha}LPA"