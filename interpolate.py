from tqdm import tqdm
import torch
import numpy as np
import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from typing import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
from model import Transformer, OnlyMLP, OnlyMLP_onlyadd, SLTHMLP
from data_module import gen_train_test, train_test_split, ArithmeticDataModule
from utils import (
    visualize_weight_distribution,
    visualize_weight,
    lines,
    full_loss,
    full_loss_mlp,
    visualize_embedding,
    get_weight_norm,
    lp_reg,
    model_inter,
)
from config.config_inter import Exp
import warnings
from pruner import Pruner, Rand, Mag, SNIP, GraSP, SynFlow
from generator import masked_parameters

warnings.filterwarnings("ignore")
import copy
import argparse


def main(config,weight_path1,weight_path2):
    wandb.init(
        project="grokking_landscape", name=f"{weight_path1}_{weight_path2}", config=config
    )
    if config.model == "transformer":
        model = Transformer(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_mlp=config.d_mlp,
            d_head=config.d_head,
            num_heads=config.num_heads,
            n_ctx=config.n_ctx,
            act_type=config.act_type,
            use_cache=False,
            use_ln=config.use_ln,
        )
    elif config.model == "mlp":
        model1 = OnlyMLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
        )
        model2 = OnlyMLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
        )
    model1.load_state_dict(torch.load(weight_path1)["model"])
    model1.to("cuda")
    model2.load_state_dict(torch.load(weight_path2)["model"])
    model2.to("cuda")
    train, test = gen_train_test(
        config.frac_train,
        config.d_vocab,
        seed=config.seed,
        is_symmetric_input=config.is_symmetric_input,
    )
    
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    run_name = f"{config.exp_name}"
    if config.save_models:
        os.makedirs(config.root / run_name, exist_ok=True)
        save_dict = {
            "model": model1.state_dict(),
            "train_data": train,
            "test_data": test,
        }
        torch.save(save_dict, config.root / run_name / "init.pth")
    train_losses = []
    test_losses = []
    alpha_range = np.arange(config.alpha_range[0], config.alpha_range[1], config.alpha_step)
    print(alpha_range)
    with tqdm(range(config.num_step)) as pbar:
        pbar.set_description(f"{run_name}")
        for iter in pbar:
            alpha = alpha_range[iter]
            #print(alpha)
            model = model_inter(model1, model2, alpha)
            model.train()
            if config.model == "transformer":
                train_loss, train_acc, train_prob = full_loss(
                    model, train, fn=config.fn, p=config.p, is_div=config.is_div
                )
                test_loss, test_acc, test_prob = full_loss(
                    model, test, fn=config.fn, p=config.p, is_div=config.is_div
                )
            elif config.model == "mlp":
                train_loss, train_acc, train_prob = full_loss_mlp(
                    model, train, config.fn, config.p, is_div=config.is_div
                )
                test_loss, test_acc, test_prob = full_loss_mlp(
                    model, test, config.fn, config.p, is_div=config.is_div
                )
            pbar.set_postfix(
                OrderedDict(
                    Train_Loss=train_loss.item(),
                    Test_Loss=test_loss.item(),
                    Train_acc=train_acc,
                    Test_acc=test_acc,
                )
            )

            l1norm, l2norm, l1mask_norm, l2mask_norm = get_weight_norm(model)
            wandb.log(
                {
                    "alpah": alpha,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_prob": train_prob,
                    "test_prob": test_prob,
                    "l1norm": l1norm,
                    "l2norm": l2norm,
                    "l1mask_norm": l1mask_norm,
                    "l2mask_norm": l2mask_norm,
                }
            )

            #train_loss.backward()
            #optimizer.step()
            # scheduler.step()
            #optimizer.zero_grad()
            if test_loss.item() < config.stopping_thresh:
                break
            if (config.save_models) and (alpha % config.save_every == 0):
                fig = visualize_weight_distribution(model)
                wandb.log({"weight_distribution": fig})
                plt.close()
                ims = visualize_weight(model)
                wandb.log({"weight": ims})
                plt.close()
                emb_img = visualize_embedding(model, p=config.p)
                wandb.log({"embedding": emb_img})
                plt.close()

                if test_loss.item() < config.stopping_thresh:
                    break
                save_dict = {
                    "model": model.state_dict(),
                   #"optimizer": optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "alpha": alpha,
                }
                torch.save(save_dict, config.root / run_name / f"{alpha}.pth")
            

        if not config.save_models:
            os.mkdir(config.root / run_name)
        save_dict = {
            "model": model.state_dict(),
            #"optimizer": optimizer.state_dict(),
            #'scheduler': scheduler.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "alpha": alpha,
        }
        torch.save(save_dict, config.root / run_name / f"final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path1", type=str, default="/home/1107/prune/mlp_0L_48D_67P_        0.4F_0.001LR_1WD_TrueS_        1WS_0.4_sparsity_0seed_finalepoch_mag_pruner/final.pth")
    parser.add_argument("--weight_path2", type=str, default="/home/1107/base/mlp_0L_48D_67P_        0.4F_0.001LR_1WD_TrueS_        1WS_0seed/final.pth")
    args = parser.parse_args()
    config = Exp()
    weight_path1 = args.weight_path1
    weight_path2 = args.weight_path2

    main(config,weight_path1,weight_path2)
