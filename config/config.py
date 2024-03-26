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
        self.fn_name = 'add'  #['add', 'subtract', 'x2xyy2','rand']'
        self.is_div = True if "only" in self.fn_name  else False
        self.frac_train = 0.5
        self.is_symmetric_input = True #False if 'subtract' in self.fn_name else True
        self.num_epochs = 10000
        self.save_models = True 
        self.save_every = 200

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 0
        self.root = Path("0120/detail") 
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
        self.lp_alpha = 0
        self.momentum = 0.9
        self.sparse_k = 0.1

        self.use_ln = False
        self.optimizer = 'sgd' # ['sgd', 'adam']

        self.random_answers = np.random.randint(low=0, high=self.p, size=(self.p, self.p))

        self.fns_dict = {'add': lambda x,y:(x+y)%self.p, }
                        # 'subtract': lambda x,y:(y-x)%self.p, 
                         #'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p, 
                         #'rand':lambda x,y:self.random_answers[x][y],\
                        #'multiply':lambda x,y:(x*y)%self.p, }
                        #'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p,\
                        #'divide':lambda x,y:(x//(y+1))%self.p,
                        #'x2y2':lambda x,y:(x**2+y**2)%self.p, }
                        #'x2xyy2x':lambda x,y:(x**2+x*y+y**2+x)%self.p,\
                        #'x3xy':lambda x,y:(x**3+x*y)%self.p, 'x3xy2y':lambda x,y:(x**2+x*y**2+y)%self.p,}
                        #'x2xyy2x2':lambda x,y:(x**2+x*y+y**2+x**2)%self.p, 'x2xyy2y2':lambda x,y:(x**2+x*y+y**2+y**2)%self.p}
        
        self.fn = self.fns_dict[self.fn_name]
        self.exp_name = f"{self.optimizer}_WD_{self.weight_decay}_LR_{self.lr}_MO_{self.momentum}"