# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import hessian
from torcheval.metrics.functional import multiclass_accuracy
import numpy as np

import seaborn as sns

sns.set()
import wandb
import math
from sklearn.decomposition import PCA #主成分分析器
import matplotlib.pyplot as plt

# %matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
import torch.autograd as autograd


from functools import *
from sklearn.manifold import TSNE
import copy


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir="fwd"):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)

        if dir == "fwd":
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir == "bwd":
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir="fwd"):
        if (dir == "fwd") or (dir == "both"):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir == "bwd") or (dir == "both"):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ["fwd", "bwd", "both"]:
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
    def __init__(self, weight_scale , weight_learning, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  # original
        # self.scores = nn.Parameter(torch.Tensor(self.weight.size(), dtype=torch.half))

        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        torch.nn.init.normal_(self.weight, mean=0, std=weight_scale / np.sqrt(self.in_features))
        # torch.nn.init.normal_(self.weight, mean=0, std=weight_scale / np.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
        # self.weight.data *= weight_scale

        self.weight_learning = weight_learning
        # NOTE: turn the gradient on the weights off
        if self.weight_learning :
            self.weight.requires_grad = True
        else:
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
        if self.weight_learning:
            w = self.weight
        else:
            w = self.weight * subnet
        x = F.conv2d(
            x,
            w,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return x

    def get_subnet(self):
        return GetSubnet.apply(self.clamped_scores, self.prune_rate)


class SupermaskLinear(nn.Linear):
    def __init__(self, weight_scale=1.0, weight_learning=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  # Original
        # self.scores = nn.Parameter(torch.Tensor(self.weight.size(), dtype=torch.half))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        # NOTE: initialize the weights like this.
        # nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.normal_(self.weight, mean=0, std=weight_scale / np.sqrt(self.in_features))
        # self.weight.data *= weight_scale
        
        self.weight_learning = weight_learning
        # NOTE: turn the gradient on the weights off
        if self.weight_learning :
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def set_prune_rate_from_threshold(self, threshold):
        k = (self.clamped_scores >= threshold).sum().item()
        n = self.scores.numel()
        self.prune_rate = k / n

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        # subnet = subnet.repeat_interleave(B,dim=0)
        if self.weight_learning:
            w = self.weight 
        else:
            w = self.weight * subnet
        return F.linear(x, w, bias=None)


class SupermaskEmbedd(nn.Linear):
    def __init__(self,  weight_scale , weight_learning, *args, **kargs):
        super().__init__(*args, **kargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        
        
        self.weight_learning = weight_learning
        # NOTE: turn the gradient on the weitghts off
        if self.weight_learning :
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    def set_prune_rate_from_threshold(self, threshold):
        k = (self.clamped_scores >= threshold).sum().item()
        n = self.scores.numel()
        self.prune_rate = k / n

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        # subnet = subnet.repeat_interleave(B,dim=0)
        if self.weight_learning:
            w = self.weight
        else:
            w = self.weight * subnet
        return torch.einsum("dbp -> bpd", w[:, x])


# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated() / 1e9)


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
    labels = torch.tensor([fn(i, j) for i, j in data]).to("cuda")
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p * 2, average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p, average="micro"
        ).item()
    return (
        cross_entropy_high_precision(logits, labels),
        accuracy,
        torch.mean(torch.gather(prob, index=labels[:, None], dim=-1)),
    )


def full_loss_multi(model, data, fn_dict, p, is_div=False, fn_names=["add", "substract"]):
    logits = model(data)[:, -1]
    prob = F.softmax(logits, dim=1)
    pred = torch.argmax(prob, dim=1)
    labels = torch.tensor([fn_dict[fn_names[op-p-1]](a, b) for a, op ,b, _ in data ]).to("cuda")
    fig = visalize_sample(pred, labels, fn_dict, fn_names, p, data)
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p * 2, average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p+1+len(fn_names), average="micro"
        ).item()
    return (
        cross_entropy_high_precision(logits, labels),
        accuracy,
        torch.mean(torch.gather(prob, index=labels[:, None], dim=-1)),
        fig
    )

def full_loss_mlp(model, data, fn, p, is_div=False, device="cuda", vis=True):
    # Take the final position only
    logits = model(data)
    prob = F.softmax(logits, dim=1)
    labels = torch.tensor([fn(i, j) for i, j in data]).to(device)
    pred = torch.argmax(prob, dim=1)
    if vis:
        fig = visalize_sample_v2(pred, labels, ["add"], p, data)
    else:
        fig = None
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p * 2, average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p, average="micro"
        ).item()
    return (
        cross_entropy_high_precision(logits, labels),
        accuracy,
        torch.mean(torch.gather(prob, index=labels[:, None], dim=-1)),
        fig
    )

def calc_hess(model, data, fn, p, is_div=False):
        # Take the final position only
    logits = model(data)
    prob = F.softmax(logits, dim=1)
    labels = torch.tensor([fn(i, j) for i, j in data]).to("cuda")
    if is_div:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p * 2, average="micro"
        ).item()
    else:
        accuracy = multiclass_accuracy(
            input=logits, target=labels, num_classes=p, average="micro"
        ).item()
    hess_loss = cross_entropy_high_precision(logits, labels)
    param = get_param(model)
    hess = hessian(cross_entropy_high_precision, param, create_graph=True)

    return hess



def to_numpy(tensor, flat=False):
    if type(tensor) != torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def line(x, y=None, hover=None, xaxis="", yaxis="", **kwargs):
    if type(y) == torch.Tensor:
        y = to_numpy(y, flat=True)
    if type(x) == torch.Tensor:
        x = to_numpy(x, flat=True)
    fig = px.line(x, y=y, hover_name=hover, **kwargs)
    fig.update_layout(xaxis_title=xaxis, yaxis_title=yaxis)
    fig.show()


def scatter(x, y, **kwargs):
    px.scatter(x=to_numpy(x, flat=True), y=to_numpy(y, flat=True), **kwargs).show()


def lines(
    lines_list,
    x=None,
    mode="lines",
    labels=None,
    xaxis="",
    yaxis="",
    title="",
    log_y=False,
    hover=None,
    **kwargs,
):
    # Helper function to plot multiple lines
    if type(lines_list) == torch.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x = np.arange(len(lines_list[0]))
    fig = go.Figure(layout={"title": title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line) == torch.Tensor:
            line = to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(
            go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs)
        )
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()


def line_marker(x, **kwargs):
    lines([x], mode="lines+markers", **kwargs)


def plot_dists(val_dict, color="C0", xlabel=None, stat="count", use_kde=True):
    columns = len(val_dict)
    fig, ax = plt.subplots(1, columns, figsize=(columns * 1, 8))
    fig_index = 0
    for key in sorted(val_dict.keys()):
        key_ax = ax[fig_index % columns]
        sns.histplot(
            val_dict[key],
            ax=key_ax,
            color=color,
            bins=50,
            stat=stat,
            kde=use_kde and ((val_dict[key].max() - val_dict[key].min()) > 1e-8),
        )  # Only plot kde if there is variance
        # key_ax.set_title(f"{key} " + (r"(%i $\to$ %i)" % (val_dict[key].shape[1], val_dict[key].shape[0]) if len(val_dict[key].shape)>1 else ""))
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
        plt.title(name, fontsize=20)
        plt.colorbar(im)
        ims.append(wandb.Image(im))
        plt.close()

    return ims


def visualize_embedding(model, data):
    #data = [(i, i) for i in range(p)]
    data = torch.tensor(data).to("cuda")
    emb = model.embed(data)
    emb = emb[:, 0, :].detach().cpu().numpy()
    emb = TSNE(n_components=2).fit_transform(emb)
    emb_dict = {}
    for ind, (i, j) in enumerate(data):
        emb_dict[i] = emb[ind, :]
        img = plt.scatter(emb[ind, 0], emb[ind, 1], c="b", alpha=0.5, s=150)
        plt.annotate(f"{i}", (emb[ind, 0], emb[ind, 1]), ha="center")
    return wandb.Image(img)

@torch.no_grad()
def visualize_neuron_activation(model, data):
    #data = [(i, i) for i in range(p)]
    fig = plt.figure(figsize=(10,10))
    data = torch.tensor(data).to("cuda")
    act = model.get_neuron_activation(data)
    act = act.detach().cpu().numpy()
    max_act = np.max(act) + np.max(act)*0.1
    act_dict = {}  
    for ind, (i, j) in enumerate(data):
        act_dict[i] = act[ind, :]
        plt.scatter(act[ind, 0], act[ind, 1], c="b", alpha=0.5, s=150)
        plt.plot([0,act[ind, 0]], [0,act[ind, 1]], c="b", alpha=0.5, linewidth=0.5)
        plt.annotate(f"{(i.item(),j.item())}", (act[ind, 0], act[ind, 1]), ha="center")
    plt.xlim(-max_act,max_act)
    plt.ylim(-max_act,max_act)        
    return fig

def visualize_neuron_activation_v2(model, data):
    #data = [(i, i) for i in range(p)]
    fig = plt.figure(figsize=(10,10))
    data = torch.tensor(data).to("cuda")
    act = model.get_neuron_activation(data)
    act = act.detach().cpu().numpy()
    #主成分分析の実行
    pca = PCA()
    pca.fit(act)
    # データを主成分空間に写像
    feature = pca.transform(act)
    act = feature
    max_act = np.max(act) + np.max(act)*0.1
    for ind, (i, j) in enumerate(data):
        plt.scatter(act[ind, 0], act[ind, 1], c="b", alpha=0.5, s=150)
        plt.plot([0,act[ind, 0]], [0,act[ind, 1]], c="b", alpha=0.5, linewidth=0.5)
        plt.annotate(f"{(i.item(),j.item())}", (act[ind, 0], act[ind, 1]), ha="center")
    plt.xlim(-max_act,max_act)
    plt.ylim(-max_act,max_act)        
    return fig

def get_weight_norm(model):
    weights = {}
    mask_keys = [k for k in model.state_dict().keys() if "weight_mask" in k]
    param_keys = [
        k for k in model.state_dict().keys() if ("mask" not in k) and ("b_" not in k)
    ]
    l2norm = 0
    l2mask_norm = 0
    l1norm = 0
    l1mask_norm = 0

    for mask_key, param_key in zip(mask_keys, param_keys):
        mask = model.state_dict()[mask_key].detach().cpu()
        param = model.state_dict()[param_key].detach().cpu()
        l2mask_norm += torch.norm(param * mask, 2)
        l2norm += torch.norm(param, 2)
        l1mask_norm += torch.norm(param * mask, 1)
        l1norm += torch.norm(param, 1)
    return (
        l1norm.item() / len(param_keys),
        l2norm.item() / len(param_keys),
        l1mask_norm.item() / len(param_keys),
        l2mask_norm.item() / len(param_keys),
    )
    
def get_weight_sparsity(model, k=0.1):
    total_weight = torch.cat([model.state_dict()[key].flatten() for key in model.state_dict().keys()], dim=0)
    weight_epsilon = total_weight.abs().max().item()*k
    weight_sparsity = torch.sum(total_weight.abs() < weight_epsilon).item()/total_weight.numel()
    return weight_sparsity

def visalize_sample(pred, labels, fn_dict, fn_names, p, data):
    fig, ax = plt.subplots(1, len(fn_names), figsize=(5*len(fn_names), 5))
    sample_acc = np.zeros((len(fn_names), p, p))
    if  len(fn_names) > 1:
        for i,(a, op, b, _) in enumerate(data):
            if pred[i] == labels[i]:
                sample_acc[op-p-1, a, b] = 1
            else:
                sample_acc[op-p-1, a, b] -= 1
        for j, fn_name in enumerate(fn_names):
            cb = ax[j].imshow(sample_acc[j], cmap="Blues", vmin=-1, vmax=1)
            ax[j].grid(False)
            ax[j].set_title(fn_name)
            ax[j].set_xlabel("b")
            ax[j].set_ylabel("a")
        fig.colorbar(cb, ax=ax.ravel().tolist(), pad=0.025)
    else:
        for i,(a, op, b, _) in enumerate(data):
            if pred[i] == labels[i]:
                sample_acc[op-p-1, a, b] = 1
            else:
                sample_acc[op-p-1, a, b] -= 1
        for j, fn_name in enumerate(fn_names):
            cb = ax.imshow(sample_acc[j], cmap="Blues", vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_title(fn_name)
            ax.set_xlabel("b")
            ax.set_ylabel("a")
        fig.colorbar(cb, ax=ax, pad=0.025)
    return fig

def visalize_sample_v2(pred, labels, fn_names, p, data):
    fig, ax = plt.subplots(1, len(fn_names), figsize=(5*len(fn_names), 5))
    sample_acc = np.zeros((len(fn_names), p, p))
    if  len(fn_names) > 1:
        for i,(a, b) in enumerate(data):
            if pred[i] == labels[i]:
                sample_acc[0, a, b] = 1
            else:
                sample_acc[0, a, b] -= 1
        for j, fn_name in enumerate(fn_names):
            cb = ax[j].imshow(sample_acc[j], cmap="Blues", vmin=-1, vmax=1)
            ax[j].grid(False)
            ax[j].set_title(fn_name)
            ax[j].set_xlabel("b")
            ax[j].set_ylabel("a")
        fig.colorbar(cb, ax=ax.ravel().tolist(), pad=0.025)
    else:
        for i,(a, b ) in enumerate(data):
            if pred[i] == labels[i]:
                sample_acc[0, a, b] = 1
            else:
                sample_acc[0, a, b] -= 1
        for j, fn_name in enumerate(fn_names):
            cb = ax.imshow(sample_acc[j], cmap="Blues", vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_title(fn_name)
            ax.set_xlabel("b")
            ax.set_ylabel("a")
        fig.colorbar(cb, ax=ax, pad=0.025)
    return fig

@torch.no_grad()
def visalize_attention(model):
    attn_matrix = model.get_attention_matrix()
    num_layer = len(attn_matrix)
    num_heads = attn_matrix[0].size(1)
    fig, ax = plt.subplots(num_heads, num_layer, figsize=(5*num_layer, 5))
    if num_layer == 1 and num_heads == 1:
        for i, attn in enumerate(attn_matrix):
            for j in range(num_heads):
                ax.imshow(attn[j].mean(dim=0).detach().cpu().numpy(), cmap="Blues")
                ax.set_title(f"Layer {i}, Head {j}")
    elif num_layer == 1:
        ax = ax[None, :]
        for i, attn in enumerate(attn_matrix):
            for j in range(num_heads):
                ax[i, j].imshow(attn[j].mean(dim=0).detach().cpu().numpy(), cmap="Blues")
                ax[i, j].set_title(f"Layer {i}, Head {j}")
    elif num_heads == 1:
        ax = ax[:, None]
        for i, attn in enumerate(attn_matrix):
            for j in range(num_heads):
                ax[i, j].imshow(attn[j].mean(dim=0).detach().cpu().numpy(), cmap="Blues")
                ax[i, j].set_title(f"Layer {i}, Head {j}")
    plt.tight_layout()
    #plt.colorbar()
    return fig

def lp_reg(model, p: int = 2):
    lp_loss = torch.tensor(0.0, requires_grad=True)
    for w in model.parameters():
        lp_loss = lp_loss + torch.norm(w, p=p)
    lp_loss = lp_loss / len(list(model.parameters()))
    return lp_loss

def model_inter(model1,model2,alpha):
    model = copy.deepcopy(model1)
    for name, param in model.named_parameters():
        param.data = alpha*model1.state_dict()[name].data + (1-alpha)*model2.state_dict()[name].data
    return model

def get_param(model):
    param = []
    for name, p in model.named_parameters():
        if "mask" not in name:
            param.append(p.flatten())
    
    return torch.cat(param)

def get_new_grads(model, x,y,current_example_index):
    '''
    Args 
        model: the model that we want to get the gradients from
        x: the input data
        y: the labels
        current_example_index: the index of the example that we want to flip
    Returns
        grads_list: a dictionary of the gradients of the model
        final_preds: the final predictions of the model
    '''
    grads_list = {}

    final_preds = model(x)

    loss = nn.CrossEntropyLoss(reduction = 'none')(final_preds, y)
    batch_size = y.shape[0]
    #for the example that we want to flip, we must reverse the loss while maintaing the population loss
    loss[current_example_index] *= -1*batch_size
    loss = loss.mean()
    loss.backward()

    for name, param in (model.named_parameters()):
        grads_list[name] = copy.deepcopy(param.grad.detach())

    # ipdb.set_trace()
    model.zero_grad()

    return grads_list, final_preds.detach()