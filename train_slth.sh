#!/usr/bin/zsh 
s=0
python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_learning
python train_slth.py -o adam -w 2 -l 0.01 -s $s --width_ratio 1 --weight_learning
python train_slth.py -o adam -w 3 -l 0.01 -s $s --width_ratio 1 --weight_learning
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 16 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 16
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 32 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 32
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 64 --weight_learning 
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 64
# # python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 128 --weight_learning                  
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 128
# # python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 256 --weight_learning
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 256


# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8192
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 16384
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 65536

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.0
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.1
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.2
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.3
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.4
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.5
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.6
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.7
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.8
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 0.9
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1  --exp_tag pruning_rate --pruning_rate 1.0