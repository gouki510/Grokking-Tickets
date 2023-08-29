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
from model import Transformer, OnlyMLP, MnistMLP
from data_module import gen_train_test, train_test_split, MNISTDataModule
from utils import visualize_weight_distribution, visualize_weight, lines, full_loss, full_loss_mlp, \
    visualize_embedding, get_weight_norm
from config_mnist import Exp
import warnings
warnings.filterwarnings("ignore")



def main(config):
    wandb.init(project="grokking_mnist",name=config.exp_name, config=config)
    model = MnistMLP(num_layers=config.num_layers, d_input=config.d_input, d_model=config.d_model, d_class=config.d_class, 
                     act_type=config.act_type,  use_ln=config.use_ln, weight_scale=config.weight_scale)
    model.to('cuda')
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    run_name = f"{config.exp_name}"
    datamodule = MNISTDataModule(config.batch_size)
    train_loader,test_loader = datamodule.get_dataloader()
    print("train_loader",len(train_loader) , "test_loader",len(test_loader))
    print("train_loader",int(len(train_loader)*config.frac_train) , "test_loader",len(test_loader))
    if config.save_models:
        os.makedirs(config.root/run_name,exist_ok=True)
        save_dict = {'model':model.state_dict()}
        torch.save(save_dict, config.root/run_name/'init.pth')
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    with tqdm(range(config.num_epochs)) as pbar:
      pbar.set_description(f'{run_name}')
      for epoch in pbar:
        loss_sum = 0
        acc_sum = 0
        for i, (inputs, labels) in enumerate(train_loader):
            if i == int(len(train_loader)*config.frac_train):
                break
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

            inputs = inputs.view(-1, config.d_input) 
            outputs = model(inputs)
            
            onehot_labels = F.one_hot(labels, num_classes=config.d_class).float()
            train_loss = criterion(outputs, onehot_labels)
            loss_sum += train_loss
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            acc_sum += acc
            train_loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss_ave = loss_sum / int(len(train_loader)*config.frac_train)
        train_acc = acc_sum / int(len(train_loader)*config.frac_train)


        model.eval()

        loss_sum = 0
        acc_sum = 0

        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                inputs = inputs.view(-1, config.d_input) 
                outputs = model(inputs)


                onehot_labels = F.one_hot(labels, num_classes=config.d_class).float()
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
                    Test_acc=test_acc
                )
            )
          
        l1norm,l2norm = get_weight_norm(model)
        wandb.log({"epoch": epoch, "train_loss": train_loss_ave, "test_loss": test_loss_ave, "train_acc":train_acc, "test_acc":test_acc, \
                    "l1norm":l1norm, "l2norm":l2norm})
        if test_loss_ave.item() < config.stopping_thresh:
              break
        if (config.save_models) and (epoch%config.save_every == 0):
            #fig = visualize_weight_distribution(model)
            #wandb.log({"weight_distribution": fig})
            #plt.close()
            ims = visualize_weight(model)
            wandb.log({"weight": ims})
            plt.close()
            #emb_img = visualize_embedding(model,p=config.p)
            #wandb.log({"embedding": emb_img})
            #plt.close()

            if test_loss_ave.item() < config.stopping_thresh:
                break
            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
                'train_loss': train_loss_ave,
                'test_loss': test_loss_ave,
                'epoch': epoch,
            }
            torch.save(save_dict, config.root/run_name/f"{epoch}.pth")
            #print(f"Saved model to {root/run_name/f'{epoch}.pth'}")
      if not config.save_models:
          os.mkdir(config.root/run_name)
      save_dict = {
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          #'scheduler': scheduler.state_dict(),
          'train_loss': train_loss_ave,
          'test_loss': test_loss_ave,
          'epoch': epoch,
      }
      torch.save(save_dict, config.root/run_name/f"final.pth")


if __name__ == '__main__':
    config = Exp()
    main(config)