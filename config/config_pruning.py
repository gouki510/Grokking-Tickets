import os
import numpy as np
from pathlib import Path

class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-3 
        self.weight_decay =  1
        self.p = 67 
        self.d_emb = 500
        self.d_model =  48
        self.fn_name = "add"#'subtract'  #['add', 'subtract', 'x2xyy2','rand']'
        self.is_div = True if "only" in self.fn_name  else False
        self.frac_train = 0.4
        self.is_symmetric_input = True
        self.num_epochs = 50000
        self.save_models = True 
        self.save_every = 50000 

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 0
        self.root = Path("1112/prune") 
        self.pre_root = Path("0927/exp1/mlp") 
        self.model = 'mlp' # ['mlp', 'transformer']
        os.makedirs(self.root,exist_ok=True)

        self.num_layers = 0
        self.batch_style = 'full' # ['full', 'random'] 
        self.d_vocab = self.p
        self.n_ctx = 2
        self.d_mlp = 1*self.d_model
        self.num_heads = 1
        assert self.d_model % self.num_heads == 0
        self.d_head = self.d_model//self.num_heads  
        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 1 #0.5576312536233431
        self.prune_rate = 0.4                               
        self.weight_ratio = -1#0.6493382079831002 #0.4152939027995708

        self.use_ln = False

        self.random_answers = np.random.randint(low=0, high=self.p, size=(self.p, self.p))

        self.fns_dict = {'add': lambda x,y:(x+y)%self.p, }
                         #'subtract': lambda x,y:(y-x)%self.p,} \
                         #'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p, 'rand':lambda x,y:self.random_answers[x][y],\
                        #'multiply':lambda x,y:(x*y)%self.p, \
                        #'x2xyy2':lambda x,y:(x**2+x*y+y**2)%self.p,}
                        #'divide':lambda x,y:(x//(y+1))%self.p,
                        #'x2y2':lambda x,y:(x**2+y**2)%self.p }
        #'x2xyy2x':lambda x,y:(x**2+x*y+y**2+x)%self.p,\
                        #'x3xy':lambda x,y:(x**3+x*y)%self.p, 'x3xy2y':lambda x,y:(x**2+x*y**2+y)%self.p,}
        self.fn = self.fns_dict[self.fn_name]
        
        # pruning
        self.pruner = "mag" # ["rand", "mag", "snip", "grasp", "synflow"]
        self.sparsity = 0.4#0.7 #0.29#0.4#0.598#1#0.3
        self.schedule = "linear" # ["linear", "exponential"]
        self.scope = "global" # ["global", "local"]
        self.epochs =  1           
        self.reinitialize =  True
        self.train_mode = False
        self.shuffle = False
        self.invert = False
        if self.is_symmetric_input:
            self.batch_size =  (self.p**2 - self.p)//2
        else:
            self.batch_size =  self.p**2
        self.if_mask_reset = False

        self.exp_name = f"{self.model}_{self.num_layers}L_{self.d_model}D_{self.p}P_\
        {self.frac_train}F_{self.lr}LR_{1}WD_{self.is_symmetric_input}S_\
        {self.weight_scale}WS_{self.seed}seed"
        
        self.checkpoint = "final"
        self.weight_path = self.pre_root/self.exp_name/f"{self.checkpoint}.pth"
        self.init_weight_path = self.pre_root/self.exp_name/"init.pth"

        self.exp_name = f"{self.model}_{self.pruner}_PR_{self.num_layers}L_{self.d_model}D_{self.p}P_\
            {self.frac_train}F_{self.lr}LR_{self.weight_decay}WD_{self.is_symmetric_input}S_\
            {self.weight_scale}WS_{self.fn_name}task_{self.reinitialize}_reinit_{self.checkpoint}_check_{self.sparsity}_sparsity"
