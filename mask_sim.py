from glob import glob
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config.config import Exp
from model import OnlyMLP
from matplotlib import cm
from typing import List

def jaccard_distance(x, y):
    inter = torch.where(x+y==2, 1, 0).sum()
    union = torch.where(x+y==0, 0, 1).sum()
    return 1 - inter/union

def main(dir_path):

    weight_paths = glob(os.path.join(dir_path,"*"))
    jaccard_distance_layer = {}
    for layer_i, layer in enumerate(["embed", "inproj", "outproj", "unembed"]):
        total_masks = {}    
        print(layer)
        for weight_path in weight_paths:
            weight_path = os.path.join(weight_path,"final.pth")
            name = weight_path.split("/")[-2].split("_")[-1].replace("epoch", "").replace("000", "k")
            if "final" in name :
                continue
            config = Exp()
            model = OnlyMLP(num_layers=config.num_layers, d_vocab=config.d_vocab, \
                                    d_model=config.d_model, d_emb=config.d_emb, \
                                    act_type=config.act_type,  use_ln=config.use_ln, \
                                    weight_scale=config.weight_scale)
            model.load_state_dict(torch.load(weight_path)["model"])
            W_mask = model.state_dict()[f"{layer}.weight_mask"]
            total_masks[name] = W_mask.view(-1)
        print(total_masks)
        jaccard_dis = torch.zeros((len(total_masks), len(total_masks)))
        for i, (name1, mask1) in enumerate(total_masks.items()):
            for j, (name2, mask2) in enumerate(total_masks.items()):
                jaccard_dis[i][j] = jaccard_distance(mask1, mask2)
        jaccard_distance_layer[layer] = jaccard_dis.numpy()
        # ax[layer_i//2][layer_i%2].imshow(jaccard_dis.numpy(), cmap=cm.Blues, interpolation='nearest')
        # ax[layer_i//2][layer_i%2].grid(False)
        # ax[layer_i//2][layer_i%2].set_xticks(np.arange(len(total_masks)))
        # ax[layer_i//2][layer_i%2].set_yticks(np.arange(len(total_masks)))
        # ax[layer_i//2][layer_i%2].set_xticklabels(total_masks.keys(), rotation=90, fontsize=20)
        # ax[layer_i//2][layer_i%2].set_yticklabels(total_masks.keys(), fontsize=20)
        # ax[layer_i//2][layer_i%2].set_title(f"{layer}", fontsize=40)
        #plt.title("Jaccard Distance", fontsize=４0)
    # colorbar
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(ax[0][0].imshow(jaccard_distance_layer["embed"], cmap=cm.Blues, interpolation='nearest'), cax=cbar_ax)
    torch.save(jaccard_distance_layer, os.path.join(dir_path, "jaccard_distance_layer.pth"))


if __name__ == "__main__":
    main("20240515/detail")