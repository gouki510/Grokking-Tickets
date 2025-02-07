#!/usr/bin/zsh 
s=0
python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_learning
python train_slth.py -o adam -w 2 -l 0.01 -s $s --width_ratio 1 --weight_learning
python train_slth.py -o adam -w 3 -l 0.01 -s $s --width_ratio 1 --weight_learning