import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd.functional import hessian
import numpy as np
import wandb
from tqdm import tqdm
from typing import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt
from model import Transformer, OnlyMLP, OnlyMLP_onlyadd, SLTHMLP, OnlyMLP_1layer
from data_module import gen_train_test, train_test_split, gen_train_test_multi
from utils import (
    visualize_weight_distribution,
    visualize_weight,
    lines,
    full_loss,
    full_loss_mlp,
    calc_hess,
    visualize_embedding,
    get_weight_norm,
    lp_reg,
    get_param,
    visualize_neuron_activation,
    visualize_neuron_activation_v2,
    get_weight_sparsity
)
from config.config import Exp
import warnings

warnings.filterwarnings("ignore")
import argparse


def main(config):
    wandb.init(project="Neurips2024_sym", name=config.exp_name, config=config)
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
            if config.is_1layer:
                model = OnlyMLP_1layer(
                    num_layers=config.num_layers,
                    d_vocab=config.d_vocab,
                    d_model=config.d_model,
                    d_emb=config.d_emb,
                    act_type=config.act_type,
                    use_ln=config.use_ln,
                    weight_scale=config.weight_scale,
                )
            else:
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
            if config.is_1layer:
                model = OnlyMLP_1layer(
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
    if config.optimizer == "adam":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98),
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    run_name = config.exp_name
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
        print(f"Saved model to {config.root/run_name/'init.pth'}")
        torch.save(save_dict, config.root / run_name / "init.pth")
    train_losses = []
    test_losses = []
    with tqdm(range(config.num_epochs)) as pbar:
        pbar.set_description()
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
                train_loss, train_acc, train_prob, train_sample = full_loss_mlp(
                    model, train, config.fn, config.p, is_div=config.is_div
                )
                """params = model.parameters()#get_param(model)
                env_grads = torch.autograd.grad(train_loss, params, create_graph=True)
                hess_train = 0
                for k ,env_grad in enumerate(env_grads):
                    for i in range(env_grad.size(0)):
                        for j in range(env_grad.size(1)):
                            grad = env_grad[i][j]
                            params = model.parameters()
                            hess_train += torch.autograd.grad(grad, params, create_graph=True)[k][i, j] 
                hess_train = hess_train / (env_grads[0].size(0)*env_grads[0].size(1)*len(env_grads))
                """

                train_loss += config.lp_alpha*lp_reg(model, config.lp)
                test_loss, test_acc, test_prob, test_sample = full_loss_mlp(
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
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_prob": train_prob,
                    "test_prob": test_prob,
                }
            )
            train_loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            if test_loss.item() < config.stopping_thresh:
                break
            if (config.save_models) and (epoch % config.save_every == 0):
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
                torch.save(save_dict, config.root / run_name / '{epoch}.pth'.format(epoch=epoch))
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
        torch.save(save_dict, config.root / run_name / "final.pth")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-o", "--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("-w", "--weight_decay", type=float, default=1, help="weight_decay")
    parser.add_argument("-l", "--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument( "--width", type=int, default=48, help="width")
    parser.add_argument( "--is_symmetric_input", action="store_true", help="is_symmetric_input")
    parser.add_argument( "--is_1layer", action="store_true", help="is_1layer")
    config = Exp()
    config.seed = parser.parse_args().seed
    config.optimizer = parser.parse_args().optimizer
    config.weight_decay = parser.parse_args().weight_decay
    config.lr = parser.parse_args().lr
    config.momentum = parser.parse_args().momentum
    config.d_model = parser.parse_args().width
    config.is_symmetric_input = parser.parse_args().is_symmetric_input
    config.exp_name = "WIDTH_{d_model}".format(d_model=config.d_model)
    config.is_1layer = parser.parse_args().is_1layer
    if config.is_1layer:
        config.exp_name += "_1layer"
    main(config)
