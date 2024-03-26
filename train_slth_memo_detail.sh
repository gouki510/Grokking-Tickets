#!/usr/bin/zsh 
s=3

python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_learning --num_epochs 200 --save_every 10 --exp_tag memo_detail
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --num_epochs 200 --save_every 10 --exp_tag memo_detail
python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4 --weight_learning --num_epochs 200 --save_every 10 --exp_tag memo_detail
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4 --num_epochs 200 --save_every 10 --exp_tag memo_detail
python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 --weight_learning --num_epochs 200 --save_every 10 --exp_tag memo_detail
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 --num_epochs 200 --save_every 10 --exp_tag memo_detail 
