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
from data_module import gen_train_test, train_test_split
from utils import (
    visualize_weight_distribution,
    visualize_weight,
    lines,
    full_loss,
    full_loss_mlp,
    visualize_embedding,
    get_weight_norm,
    lp_reg,
)
from config.config import Exp
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(config):
    wandb.init(project="grokking_lp_reg", name=config.exp_name, config=config)
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
        if config.is_div:
            model = OnlyMLP_onlyadd(
                num_layers=config.num_layers,
                d_vocab=config.d_vocab,
                d_model=config.d_model,
                d_emb=config.d_emb,
                act_type=config.act_type,
                use_ln=config.use_ln,
                weight_scale=config.weight_scale,
            )
        else:
            model = OnlyMLP(
                num_layers=config.num_layers,
                d_vocab=config.d_vocab,
                d_model=config.d_model,
                d_emb=config.d_emb,
                act_type=config.act_type,
                use_ln=config.use_ln,
                weight_scale=config.weight_scale,
            )
    elif config.model == "slthmlp":
        model = SLTHMLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
            prune_rate=config.prune_rate,
        )
    model.to("cuda")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
    )
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    run_name = f"{config.exp_name}"
    train, test = gen_train_test(
        config.frac_train,
        config.d_vocab,
        seed=config.seed,
        is_symmetric_input=config.is_symmetric_input,
    )
    if config.save_models:
        os.makedirs(config.root / run_name, exist_ok=True)
        save_dict = {
            "model": model.state_dict(),
            "train_data": train,
            "test_data": test,
        }
        torch.save(save_dict, config.root / run_name / "init.pth")
    train_losses = []
    test_losses = []
    with tqdm(range(config.num_epochs)) as pbar:
        pbar.set_description(f"{run_name}")
        for epoch in pbar:
            if config.model == "transformer":
                train_loss, train_acc, train_prob = full_loss(
                    model, train, fn=config.fn, p=config.p, is_div=config.is_div
                )
                train_loss += config.lp_alpha*lp_reg(model, config.lp)
                test_loss, test_acc, test_prob = full_loss(
                    model, test, fn=config.fn, p=config.p, is_div=config.is_div
                )
            elif config.model == "mlp":
                train_loss, train_acc, train_prob = full_loss_mlp(
                    model, train, config.fn, config.p, is_div=config.is_div
                )
                train_loss += config.lp_alpha*lp_reg(model, config.lp)
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
                    "epoch": epoch,
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
            train_loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            if test_loss.item() < config.stopping_thresh:
                break
            if (config.save_models) and (epoch % config.save_every == 0):
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
                    "optimizer": optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict(),
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "epoch": epoch,
                }
                torch.save(save_dict, config.root / run_name / f"{epoch}.pth")
                # print(f"Saved model to {root/run_name/f'{epoch}.pth'}")
        if not config.save_models:
            os.mkdir(config.root / run_name)
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            #'scheduler': scheduler.state_dict(),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "epoch": epoch,
        }
        torch.save(save_dict, config.root / run_name / f"final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
    config = Exp()
    config.seed = parser.parse_args().seed
    config.exp_name = f"{config.model}_{config.num_layers}L_{config.d_model}D_{config.p}P_\
        {config.frac_train}F_{config.lr}LR_{config.weight_decay}WD_{config.is_symmetric_input}S_\
        {config.weight_scale}WS_{config.seed}seed"
    main(config)
