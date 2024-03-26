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
from model import Transformer, OnlyMLP, MnistMLP, SLTHMLP, SLTHCNN
from data_module import gen_train_test, train_test_split, MNISTDataModule
from utils import (
    visualize_weight_distribution,
    visualize_weight,
    lines,
    full_loss,
    full_loss_mlp,
    visualize_embedding,
    get_weight_norm,
    get_weight_sparsity
)
from config.config_mnist_slth import Exp
import warnings
import argparse

warnings.filterwarnings("ignore")


def main(config):
    wandb.init(project="grokking_mnist_slth2", name=config.exp_name, config=config)
    if config.model == "mlp":
        model = SLTHMLP(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
            prune_rate=config.prune_rate,
            weight_learning=config.weight_learning,
            img_size=config.img_size,
        )
    elif config.model == "cnn":
        model = SLTHCNN(
            num_layers=config.num_layers,
            d_vocab=config.d_vocab,
            d_model=config.d_model,
            d_emb=config.d_emb,
            act_type=config.act_type,
            use_ln=config.use_ln,
            weight_scale=config.weight_scale,
            prune_rate=config.prune_rate,
            weight_learning=config.weight_learning,
            img_size=config.img_size,
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
    datamodule = MNISTDataModule(config.batch_size)
    train_loader, test_loader = datamodule.get_dataloader()
    print("train_loader", len(train_loader), "test_loader", len(test_loader))
    print(
        "train_loader",
        int(len(train_loader) * config.frac_train),
        "test_loader",
        int(len(test_loader) * config.frac_train),
    )
    if config.save_models:
        os.makedirs(config.root / run_name, exist_ok=True)
        save_dict = {"model": model.state_dict()}
        torch.save(save_dict, config.root / run_name / "init.pth")
    if config.criterion == "mse":
        criterion = nn.MSELoss()
    elif config.criterion == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    with tqdm(range(config.num_epochs)) as pbar:
        pbar.set_description(f"")
        for epoch in pbar:
            loss_sum = 0
            acc_sum = 0
            for i, (inputs, labels) in enumerate(train_loader):
                if i == int(len(train_loader) * config.frac_train):
                    break
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")

                if config.model == "mlp":
                    inputs = inputs.view(-1, config.img_size[0] * config.img_size[1]* config.img_size[2])
                outputs = model(inputs)

                onehot_labels = F.one_hot(labels, num_classes=config.d_vocab).float()
                train_loss = criterion(outputs, onehot_labels)
                loss_sum += train_loss
                acc = (outputs.argmax(dim=1) == labels).float().mean()
                acc_sum += acc
                train_loss.mean().backward()
                optimizer.step()
                optimizer.zero_grad()

            train_loss_ave = loss_sum / int(len(train_loader) * config.frac_train)
            train_acc = acc_sum / int(len(train_loader) * config.frac_train)

            model.eval()

            loss_sum = 0
            acc_sum = 0

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(test_loader):
                    if i == int(len(test_loader) * config.frac_train):
                        break
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                    if config.model == "mlp":
                        inputs = inputs.view(-1, config.img_size[0] * config.img_size[1]* config.img_size[2])
                    outputs = model(inputs)

                    onehot_labels = F.one_hot(
                        labels, num_classes=config.d_vocab
                    ).float()
                    loss_sum += criterion(outputs, onehot_labels)
                    acc = (outputs.argmax(dim=1) == labels).float().mean()
                    acc_sum += acc

                test_loss_ave = loss_sum / len(test_loader)
                test_acc = acc_sum / len(test_loader)

            pbar.set_postfix(
                OrderedDict(
                    Train_Loss=train_loss_ave.item(),
                    Test_Loss=test_loss_ave.item(),
                    Train_acc=train_acc,
                    Test_acc=test_acc,
                )
            )

            # l1norm, l2norm, l1mask_norm, l2mask_norm = get_weight_norm(model)
            sparsity = get_weight_sparsity(model, k=config.sparse_k)
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss_ave,
                    "test_loss": test_loss_ave,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    # "l1norm": l1norm,
                    # "l2norm": l2norm,
                    # "l1mask_norm": l1mask_norm,
                    # "l2mask_norm": l2mask_norm,
                    "sparsity": sparsity,
                }
            )
            if test_loss_ave.item() < config.stopping_thresh:
                break
            if (config.save_models) and (epoch % config.save_every == 0):
                # fig = visualize_weight_distribution(model)
                # wandb.log({"weight_distribution": fig})
                # plt.close()
                # ims = visualize_weight(model)
                # wandb.log({"weight": ims})
                # plt.close()
                # emb_img = visualize_embedding(model,p=config.p)
                # wandb.log({"embedding": emb_img})
                # plt.close()

                if test_loss_ave.item() < config.stopping_thresh:
                    break
                save_dict = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict(),
                    "train_loss": train_loss_ave,
                    "test_loss": test_loss_ave,
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
            "train_loss": train_loss_ave,
            "test_loss": test_loss_ave,
            "epoch": epoch,
        }
        torch.save(save_dict, config.root / run_name / f"final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=0, help="seed")
    parser.add_argument("-o", "--optimizer", type=str, default="sgd", help="optimizer")
    parser.add_argument("-w", "--weight_decay", type=float, default=1, help="weight_decay")
    parser.add_argument("-l", "--lr", type=float, default=1e-2, help="lr")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument( "--width_ratio", type=int, default=48, help="width")
    parser.add_argument( "--weight_learning", action="store_true", help="weight_learning")
    parser.add_argument( "--weight_init_path", default=None, help="weight_init_path")
    parser.add_argument( "--num_epochs", type=int, default=10000, help="epochs")
    parser.add_argument( "--save_every", type=int, default=200, help="save_every")
    parser.add_argument( "--exp_tag", type=str, default="normal", help="exp_tag")
    parser.add_argument( "--model", type=str, default="mlp", help="model")
    parser.add_argument( "--criterion", type=str, default="mse", help="criterion")
    
    config = Exp()
    config.seed = parser.parse_args().seed
    config.optimizer = parser.parse_args().optimizer
    config.weight_decay = parser.parse_args().weight_decay
    config.lr = parser.parse_args().lr
    config.momentum = parser.parse_args().momentum
    config.d_model = parser.parse_args().width_ratio * config.d_model
    config.d_emb = parser.parse_args().width_ratio * config.d_emb
    config.weight_learning = parser.parse_args().weight_learning
    config.weight_init_path = parser.parse_args().weight_init_path
    config.num_epochs = parser.parse_args().num_epochs
    config.save_every = parser.parse_args().save_every
    exp_tag = parser.parse_args().exp_tag
    config.model = parser.parse_args().model    
    config.criterion = parser.parse_args().criterion
    config.num_epochs = parser.parse_args().num_epochs
    
    if config.weight_init_path is not None:
        config.exp_name = "{exp_tag}/WIDTHRATIO_{ratio}_WEIGHTLR_{weight_lr}_WEIGHTINIT_{weight_init_path}".format(exp_tag=exp_tag ,ratio=parser.parse_args().width_ratio, weight_lr=config.weight_learning, weight_init_path=config.weight_init_path.replace("/", "_"))
    else:
        config.exp_name = "{exp_tag}/MODEL_{model}_WIDTHRATIO_{ratio}_WEIGHTLR_{weight_lr}".format(model=config.model,exp_tag=exp_tag,ratio=parser.parse_args().width_ratio, weight_lr=config.weight_learning)
    main(config)
