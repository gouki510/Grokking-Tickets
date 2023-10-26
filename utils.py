
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_accuracy
import numpy as np
import einops
from tqdm import tqdm

import seaborn as sns
sns.set()
import wandb
import math

import matplotlib.pyplot as plt
#%matplotlib inline
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import torch.autograd as autograd


from functools import *
import pandas as pd
from sklearn.manifold import TSNE


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size())) # original
        # self.scores = nn.Parameter(torch.Tensor(self.weight.size(), dtype=torch.half))

        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.2)

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

        # NOTE: Ensure that optimizer gets an empty parameter list by enablng this
        # self.scores.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups
        )
        return x

    def get_subnet(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate)


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size())) # Original
        # self.scores = nn.Parameter(torch.Tensor(self.weight.size(), dtype=torch.half))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def set_prune_rate_from_threshold(self, threshold):
        k = (self.clamped_scores >= threshold).sum().item()
        n = self.scores.numel()
        self.prune_rate =  k / n

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        B,f = x.size()
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        #subnet = subnet.repeat_interleave(B,dim=0)
        w = self.weight * subnet
        return F.linear(x, w, bias=None)

class SupermaskEmbedd(nn.Linear):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weitghts off
        self.weight.requires_grad = False
    
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
    
    def set_prune_rate_from_threshold(self, threshold):
        k = (self.clamped_scores >= threshold).sum().item()
        n = self.scores.numel()
        self.prune_rate =  k / n
    
    @property
    def clamped_scores(self):
        return self.scores.abs()
    
    def forward(self, x):
        B,f = x.size()
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        #subnet = subnet.repeat_interleave(B,dim=0)
        w = self.weight * subnet
        return   torch.einsum('dbp -> bpd', w[:, x])
    
    

# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated()/1e9)

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss

def full_loss(model, data, fn, p, is_div=False):
    logits = model(data)[:, -1]
    prob = F.softmax(logits, dim=1)
    labels = torch.tensor([fn(i, j) for i, j in data]).to('cuda')
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits,
            target=labels,
            num_classes=p*2,
            average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits,
            target=labels,
            num_classes=p,
            average="micro"
        ).item()
    return cross_entropy_high_precision(logits, labels), accuracy, torch.mean(torch.gather(prob, index=labels[:, None], dim=-1))

def full_loss_mlp(model, data, fn, p, is_div=False):
    # Take the final position only
    logits = model(data)
    prob = F.softmax(logits, dim=1)
    labels = torch.tensor([fn(i, j) for i, j in data]).to('cuda')
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits,
            target=labels,
            num_classes=p*2,
            average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits,
            target=labels,
            num_classes=p,
            average="micro"
        ).item()
    return cross_entropy_high_precision(logits, labels), accuracy, torch.mean(torch.gather(prob, index=labels[:, None], dim=-1))

def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()

def line(x, y=None, hover=None, xaxis='', yaxis='', **kwargs):
    if type(y)==torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x)==torch.Tensor:
        x=to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()

def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()

def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def line_marker(x, **kwargs):
    lines([x], mode='lines+markers', **kwargs)

def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns*1, 8))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index%columns]
        sns.histplot(val_dict[key], ax=key_ax, color=color, bins=50, stat=stat,
                     kde=use_kde and ((val_dict[key].max()-val_dict[key].min())>1e-8)) # Only plot kde if there is variance
        #key_ax.set_title(f"{key} " + (r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape)>1 else ""))
        key_ax.set_xlabel(key)
        fig_index += 1
    return fig

def visualize_weight_distribution(model, color="C0"):
    weights = {}
    for name, param in model.state_dict().items():
        key_name = f"Layer {name.split('.')[-1]}"
        weights[name] = param.detach().view(-1).cpu().numpy().astype(np.float32)
    fig = plot_dists(weights, color=color, xlabel="Weight vals")
    return fig


def visualize_weight(model):
    ims = []
    for name, param in model.state_dict().items():
        key_name = f"Layer {name.split('.')[-1]}"
        if len(param.shape) != 2:
          continue
        weight = param.detach().cpu().numpy().astype(np.float32)
        im = plt.pcolor(weight)
        plt.title(name,fontsize=20)
        plt.colorbar(im)
        ims.append(wandb.Image(im))
        plt.close()

    return ims

def visualize_embedding(model, p):
    data = [(i, i) for i in range(p)]
    data = torch.tensor(data).to('cuda')
    emb = model.embed(data)
    emb = emb[:,0,:].detach().cpu().numpy()
    emb = TSNE(n_components=2).fit_transform(emb)
    emb_dict = {}
    for ind,(i,j) in enumerate(data):
        emb_dict[i] = emb[ind,:]
        img = plt.scatter(emb[ind,0],emb[ind,1],c='b',alpha=0.5,s=150)
        plt.annotate(f"{i}", (emb[ind,0],emb[ind,1]),ha='center')
    return wandb.Image(img)

def get_weight_norm(model):
    weights = {}
    mask_keys = [k for k in model.state_dict().keys() if "weight_mask" in k]
    param_keys = [k for k in model.state_dict().keys() if ("mask"  not in k) and ("b_" not in k)]
    l2norm = 0
    l2mask_norm = 0
    l1norm = 0
    l1mask_norm = 0

    for mask_key, param_key in zip(mask_keys,param_keys):
        mask = model.state_dict()[mask_key].detach().cpu()
        param = model.state_dict()[param_key].detach().cpu()
        l2mask_norm += torch.norm(param*mask,2)
        l2norm += torch.norm(param,2)
        l1mask_norm += torch.norm(param*mask,1)
        l1norm += torch.norm(param,1)
    return l1norm.item()/len(param_keys), l2norm.item()/len(param_keys), l1mask_norm.item()/len(param_keys), l2mask_norm.item()/len(param_keys)

