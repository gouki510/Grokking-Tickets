import os
import numpy as np
from pathlib import Path

class Exp(object):
    def __init__(self) -> None:
        # Learning Parameter
        self.lr=1e-3 
        self.weight_decay = 0.1
        self.d_input = 784
        self.d_class = 10
        self.d_model =  200
        self.frac_train = 0.1
        self.is_symmetric_input = False
        self.num_epochs = 100000
        self.save_models = True 
        self.save_every = 1000 

        # Stop training when test loss is <stopping_thresh
        self.stopping_thresh = -1
        self.seed = 0 
        self.root = Path("some_exp") 
        self.model = 'mlp' # ['mlp', 'transformer']
        os.makedirs(self.root,exist_ok=True)

        self.num_layers = 1
        self.batch_size = 1000
        self.d_mlp = 1*self.d_model
        self.act_type = 'ReLU'  # ['ReLU', 'GELU']
        self.weight_scale = 8

        self.use_ln = False

        # pruning
        self.pruner = "mag" # ["rand", "mag", "snip", "grasp", "synflow"]
        self.sparsity = 0.4
        self.schedule = "linear" # ["linear", "exponential"]
        self.scope = "global" # ["global", "local"]
        self.epochs = 1             
        self.reinitialize =  False
        self.train_mode = False
        self.shuffle = False
        self.invert = False


        self.weight_path = "/home/0822/Mnist_mlp_1L_200D__        0.1F_0.001LR_0.1WD_FalseS_        8WS/final.pth"
        self.init_weight_path = "/home/0822/Mnist_mlp_1L_200D__        0.1F_0.001LR_0.1WD_FalseS_        8WS/init.pth"

        
        self.exp_name = f"Mnist_{self.model}_{self.num_layers}L_{self.d_model}D__\
        {self.frac_train}F_{self.lr}LR_{self.weight_decay}WD_{self.is_symmetric_input}S_\
        {self.weight_scale}WS"
