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
from utils import visualize_weight_distribution, visualize_weight, lines, full_loss, full_loss_mlp, \
    visualize_embedding, get_weight_norm
from config_pruning import Exp
import warnings
from pruner import Pruner, Rand, Mag, SNIP, GraSP, SynFlow
from generator import  masked_parameters
warnings.filterwarnings("ignore")
import copy

def prune_loop(model, loss, pruner, dataloader, device, sparsity=0.4, schedule="linear", scope="global", epochs=1,
               reinitialize=False, train_mode=False, shuffle=False, invert=False):
    """
    Prunes model according to pruner and returns masked parameters.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)
    
    # Reainitialize weights
    if reinitialize:
        masked_model = copy.deepcopy(model)
        model.load_state_dict(torch.load(config.init_weight_path)["model"])
        for (n1,m1),(n2,m2) in zip(masked_model.named_buffers(),model.named_buffers()):
            m2.copy_(m1)

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    if np.abs(remaining_params - total_params*sparsity) >= 5:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()
    
    #return pruner.masked_parameters

def main(config):
    wandb.init(project="grokking_pruning",name=config.exp_name, config=config)
    if config.model == 'transformer':
        model = Transformer(num_layers=config.num_layers, d_vocab=config.d_vocab, d_model=config.d_model, d_mlp=config.d_mlp, \
                            d_head=config.d_head, num_heads=config.num_heads, n_ctx=config.n_ctx, act_type=config.act_type, use_cache=False, use_ln=config.use_ln)
    elif config.model == 'mlp':
        model = OnlyMLP(num_layers=config.num_layers, d_vocab=config.d_vocab, \
                        d_model=config.d_model, d_emb=config.d_emb, \
                        act_type=config.act_type,  use_ln=config.use_ln, \
                        weight_scale=config.weight_scale)
    model.load_state_dict(torch.load(config.weight_path)["model"])
    model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    if config.pruner == "rand":
        pruner = Rand(masked_parameters(model))
    elif config.pruner == "mag": 
        pruner = Mag(masked_parameters(model))
    elif config.pruner == "snip":
        pruner = SNIP(masked_parameters(model))
    elif config.pruner == "grasp":
        pruner = GraSP(masked_parameters(model))
    elif config.pruner == "synflow":
        pruner = SynFlow(masked_parameters(model))
    else:
        pruner = Rand(masked_parameters(model))
    train,test = gen_train_test(config.frac_train, config.d_vocab, seed=config.seed, is_symmetric_input=config.is_symmetric_input)
    data_module = ArithmeticDataModule(train,test,config.fn,config.batch_size)
    train_dataloader, test_dataloader = data_module.get_dataloader()
    prune_loop(model, criterion, pruner, train_dataloader, 'cuda', sparsity=config.sparsity, schedule=config.schedule, scope=config.scope,\
                epochs=config.epochs, reinitialize=config.reinitialize, train_mode=config.train_mode, shuffle=config.shuffle, invert=config.invert)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    run_name = f"{config.exp_name}"
    model.train()
    if config.save_models:
        os.makedirs(config.root/run_name,exist_ok=True)
        save_dict = {'model':model.state_dict(), 'train_data':train, 'test_data':test}
        torch.save(save_dict, config.root/run_name/'init.pth')
    train_losses = []
    test_losses = []

    with tqdm(range(config.num_epochs)) as pbar:
      pbar.set_description(f'{run_name}')
      for epoch in pbar:
          if config.model == 'transformer':
            train_loss,train_acc,train_prob = full_loss(model, train, fn=config.fn , p=config.p, is_div=config.is_div)
            test_loss,test_acc,test_prob = full_loss(model, test, fn=config.fn , p=config.p, is_div=config.is_div)
          elif config.model == 'mlp':
            train_loss,train_acc,train_prob = full_loss_mlp(model, train, config.fn, config.p, is_div=config.is_div)
            test_loss,test_acc,test_prob = full_loss_mlp(model, test, config.fn, config.p, is_div=config.is_div)
          pbar.set_postfix(
                   OrderedDict(
                       Train_Loss=train_loss.item(),
                       Test_Loss=test_loss.item(),
                       Train_acc=train_acc,
                       Test_acc=test_acc
                   )
               )
          
          l1norm,l2norm = get_weight_norm(model)
          wandb.log({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "train_acc":train_acc, "test_acc":test_acc, \
                     "train_prob":train_prob, "test_prob":test_prob, "l1norm":l1norm, "l2norm":l2norm})
          train_loss.backward()
          optimizer.step()
          #scheduler.step()
          optimizer.zero_grad()
          if test_loss.item() < config.stopping_thresh:
              break
          if (config.save_models) and (epoch%config.save_every == 0):
              fig = visualize_weight_distribution(model)
              wandb.log({"weight_distribution": fig})
              plt.close()
              ims = visualize_weight(model)
              wandb.log({"weight": ims})
              plt.close()
              emb_img = visualize_embedding(model,p=config.p)
              wandb.log({"embedding": emb_img})
              plt.close()

              if test_loss.item() < config.stopping_thresh:
                  break
              save_dict = {
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  #'scheduler': scheduler.state_dict(),
                  'train_loss': train_loss,
                  'test_loss': test_loss,
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
          'train_loss': train_loss,
          'test_loss': test_loss,
          'train_losses': train_losses,
          'test_losses': test_losses,
          'epoch': epoch,
      }
      torch.save(save_dict, config.root/run_name/f"final.pth")

if __name__ == "__main__":
    config = Exp()
    main(config)