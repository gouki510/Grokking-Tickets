#!/usr/bin/zsh 
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 1 --width 48 
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 10 --width 48 
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 100 --width 48 --is_symmetric_input
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 1000 --width 48 --is_symmetric_input
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 2000 --width 48 --is_symmetric_input
# CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 4000 --width 48 --is_symmetric_input
CUDA_VISIBLE_DEVICES=1 python train_batch.py --batch_size 500 --width 48 --is_symmetric_input





