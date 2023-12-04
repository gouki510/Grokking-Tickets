import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import multiclass_accuracy
import numpy as np
import einops
import tqdm.notebook as tqdm
from utils import HookPoint
import seaborn as sns
sns.set()
from utils import GetSubnet, SupermaskLinear, SupermaskEmbedd



# Define network architecture
# I defined my own transformer from scratch so I'd fully understand each component
# - I expect this wasn't necessary or particularly important, and a bunch of this
# replicates existing PyTorch functionality

"""
 b : batch size
 d : embedding size of token
 p : vocabraly size
 i : number of heads
 h : embedding size of each heads
"""

weith_ratio = 0.5576312536233431

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_emb, weight_scale=1):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_emb, d_vocab))
        torch.nn.init.normal_(self.W_E, mean=0, std=weight_scale/np.sqrt(d_vocab))
        #nn.init.constant_(self.W_E, weight_scale)
        self.register_buffer('weight_mask', torch.ones(self.W_E.shape))
    
    def set_weight_ratio(self, weight_ratio):
        self.W_E = nn.Parameter(self.W_E * weight_ratio)
    
    def set_weight_ratio_l2(self, weight_ratio):
        self.W_E = nn.Parameter(self.W_E * torch.sqrt(weight_ratio))

    def forward(self, x):
        W = self.weight_mask * self.W_E
        #onehot_x = F.one_hot(x, num_classes=self.W_E.shape[-1]).float()
        return torch.einsum('dbp -> bpd', W[:, x])

class Unembed(nn.Module):
    def __init__(self, d_vocab, d_emb, weight_scale=1):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_emb, d_vocab))
        torch.nn.init.normal_(self.W_U, mean=0, std=weight_scale/np.sqrt(d_emb))
        #nn.init.constant_(self.W_U, weight_scale)
        self.register_buffer('weight_mask', torch.ones(self.W_U.shape))

    def set_weight_ratio(self, weight_ratio):
        self.W_U = nn.Parameter(self.W_U * weight_ratio)

    def set_weight_ratio_l2(self, weight_ratio):
        self.W_U = nn.Parameter(self.W_U * torch.sqrt(weight_ratio))

    def forward(self, x):
        W = self.weight_mask * self.W_U
        return (x @ W)

# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model, weight_scale=1):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model)*weight_scale)

    def forward(self, x):
        return x+self.W_pos[:x.shape[-2]]

# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x

# Attention
class Attention(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size (113 or 3)
    i : number of heads
    h : embedding size of each heads
    n_ctx : token size
    """
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model)/np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads)/np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.register_buffer('weight_maskK', torch.ones(self.W_K.shape))
        self.register_buffer('weight_maskQ', torch.ones(self.W_Q.shape))
        self.register_buffer('weight_maskV', torch.ones(self.W_V.shape))
        self.register_buffer('weight_maskO', torch.ones(self.W_O.shape))
    
    def set_weight_ratio(self, weight_ratio):
        self.W_K = nn.Parameter(self.W_K * weight_ratio)
        self.W_Q = nn.Parameter(self.W_Q * weight_ratio)
        self.W_V = nn.Parameter(self.W_V * weight_ratio)
        self.W_O = nn.Parameter(self.W_O * weight_ratio)


    def forward(self, x):
        W_K = self.W_K*self.weight_maskK
        W_Q = self.W_Q*self.weight_maskQ
        W_V = self.W_V*self.weight_maskV
        W_O = self.W_O*self.weight_maskO
        k = torch.einsum('ihd,bpd->biph', W_K, x)
        q = torch.einsum('ihd,bpd->biph', W_Q, x)
        v = torch.einsum('ihd,bpd->biph', W_V, x)
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked/np.sqrt(self.d_head), dim=-1)
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', W_O, z_flat)
        return out

# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_in, d_out, act_type, weight_scale=1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in))
        torch.nn.init.normal_(self.W, mean=0, std=weight_scale/np.sqrt(d_in))
        self.register_buffer('weight_mask', torch.ones(self.W.shape))
    
    def set_weight_ratio(self, weight_ratio):
        self.W = nn.Parameter(self.W * weight_ratio)

    def set_weight_ratio_l2(self, weight_ratio):
        self.W = nn.Parameter(self.W * torch.sqrt(weight_ratio))
    
    def forward(self, x):
        W = self.weight_mask * self.W
        return x @ W.T  

# for transformer
class MLP2(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size (114 or 3)
    i : number of heads
    h : embedding size of each heads
    """
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model)/np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp)/np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']
        self.register_buffer('weight_mask_in', torch.ones(self.W_in.shape))
        self.register_buffer('weight_mask_out', torch.ones(self.W_out.shape))
    
    def set_weight_ratio(self, weight_ratio):
        self.W_in = nn.Parameter(self.W_in * weight_ratio)
        self.W_out = nn.Parameter(self.W_out * weight_ratio)

    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in*self.weight_mask_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out*self.weight_mask_out, x) + self.b_out
        return x

# Transformer Block
class TransformerBlock(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size
    i : number of heads
    h : embedding size of each heads
    """
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.mlp = MLP2(d_model, d_mlp, act_type, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()
    
    def set_weight_ratio(self, weight_ratio):
        self.attn.set_weight_ratio(weight_ratio)
        self.mlp.set_weight_ratio(weight_ratio)

    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp((x))))
        return x

# Full transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, use_cache=False, use_ln=True):
        super().__init__()
        self.cache = {}
        self.use_cache = use_cache

        self.embed = Embed(d_vocab, d_model)
        #self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        self.unembed = Unembed(d_vocab, d_model)
        self.use_ln = use_ln

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)

    def set_weight_ratio(self, weight_ratio):
        self.embed.set_weight_ratio(weight_ratio)
        self.unembed.set_weight_ratio(weight_ratio)
        for block in self.blocks:
            block.attn.set_weight_ratio(weight_ratio)
            block.mlp.set_weight_ratio(weight_ratio)

    def forward(self, x):
        x = self.embed(x)
        #x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x[:,-1,:].unsqueeze(1)

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')

    def get_subnet(self,mask_dic):
        for k,v in self.state_dict().items():
          self.state_dict()[k][0:v.shape[0]] = v * mask_dic[k].to(v.device)


class OnlyMLP(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size
    i : number of heads
    h : embedding size of each heads
    """
    def __init__(self, num_layers, d_vocab, d_model, d_emb, act_type,use_ln=True, weight_scale=1):
        super().__init__()
        #self.model = [model]
        self.use_ln = use_ln
        self.unembed = Unembed(d_vocab, d_emb, weight_scale=weight_scale)
        self.embed = Embed(d_vocab, d_emb, weight_scale=weight_scale)
        self.inproj = MLP(d_emb, d_model, act_type, weight_scale=weight_scale)
        self.outproj = MLP(d_model, d_emb, act_type, weight_scale=weight_scale)
        self.mlps = nn.ModuleList([MLP(d_model,d_model,act_type,weight_scale=weight_scale) for i in range(num_layers)])
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']
        self.act = nn.ReLU() if act_type=='ReLU' else nn.GELU()
    
    def set_weight_ratio(self, weight_ratio):
        self.embed.set_weight_ratio(weight_ratio)
        self.unembed.set_weight_ratio(weight_ratio)
        self.inproj.set_weight_ratio(weight_ratio)
        self.outproj.set_weight_ratio(weight_ratio)
        for mlp in self.mlps:
            mlp.set_weight_ratio(weight_ratio)

    def forward(self, x):
        x = self.embed(x)
        x = self.inproj(x)
        x = x.sum(dim=1)
        x = self.act(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = self.act(x)
        x = self.outproj(x)
        x = self.unembed(x)
        return x

    def get_embedding(self, x):
        return self.embed(x)

class OnlyMLP_onlyadd(nn.Module):
    """
    b : batch size
    d : embedding size of token
    p : vocabraly size
    i : number of heads
    h : embedding size of each heads
    """
    def __init__(self, num_layers, d_vocab, d_model, d_emb, act_type,use_ln=True, weight_scale=1):
        super().__init__()
        #self.model = [model]
        self.use_ln = use_ln
        self.unembed = Unembed(d_vocab*2, d_emb, weight_scale=weight_scale)
        self.embed = Embed(d_vocab, d_emb, weight_scale=weight_scale)
        self.inproj = MLP(d_emb, d_model, act_type, weight_scale=weight_scale)
        self.outproj = MLP(d_model, d_emb, act_type, weight_scale=weight_scale)
        self.mlps = nn.ModuleList([MLP(d_model,d_model,act_type,weight_scale=weight_scale) for i in range(num_layers)])
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']
        self.act = nn.ReLU() if act_type=='ReLU' else nn.GELU()

    def forward(self, x):
        x = self.embed(x)
        x = self.inproj(x)
        x = x.sum(dim=1)
        x = self.act(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = self.act(x)
        x = self.outproj(x)
        x = self.unembed(x)
        return x

    def get_embedding(self, x):
        return self.embed(x)

class MnistMLP(nn.Module):
    def __init__(self, num_layers, d_input, d_model, d_class, act_type, use_ln=True, weight_scale=1):
        super().__init__()
        #self.model = [model]
        self.use_ln = use_ln
        self.inproj = MLP(d_input, d_model, act_type, weight_scale=weight_scale)
        self.outproj = MLP(d_model, d_class, act_type, weight_scale=weight_scale)
        self.mlps = nn.ModuleList([MLP(d_model,d_model,act_type,weight_scale=weight_scale) for i in range(num_layers)])
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']
        self.act = nn.ReLU() if act_type=='ReLU' else nn.GELU()

    def forward(self, x):
        x = self.inproj(x)
        x = self.act(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = self.act(x)
        x = self.outproj(x)
        return x

    def get_embedding(self, x):
        return self.embed(x)
    

class SLTHMLP(nn.Module):
    def __init__(self, num_layers, d_vocab, d_model, d_emb, act_type,use_ln=True, weight_scale=1, prune_rate=0.4):
        super().__init__()
        #self.model = [model]
        self.embbed = SupermaskEmbedd(d_vocab, d_emb, weight_scale=weight_scale)
        self.embbed.set_prune_rate(prune_rate)
        self.inproj = SupermaskLinear(d_emb, d_model, weight_scale=weight_scale)
        self.inproj.set_prune_rate(prune_rate)
        self.outproj = SupermaskLinear(d_model, d_emb, weight_scale=weight_scale)
        self.outproj.set_prune_rate(prune_rate)
        self.unembed = SupermaskLinear(d_emb, d_vocab, weight_scale=weight_scale)
        self.unembed.set_prune_rate(prune_rate)
        self.mlps = nn.ModuleList([SupermaskLinear(d_model,d_model,weight_scale=weight_scale) for i in range(num_layers)])
        for mlp in self.mlps:
            mlp.set_prune_rate(prune_rate)
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']
        self.act = nn.ReLU() if act_type=='ReLU' else nn.GELU()
    
    def forward(self, x):
        x = self.inproj(x)
        x = self.act(x)
        for mlp in self.mlps:
            x = mlp(x)
            x = self.act(x)
        x = self.outproj(x)
        return x    