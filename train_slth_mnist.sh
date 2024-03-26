#!/usr/bin/zsh 
s=0

# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 1 --weight_learning 
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 1 --weight_learning  --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.01 -l 0.01 -s $s --width_ratio 1 --weight_learning
# python train_mnist_slth.py -o adam -w 0.01 -l 0.01 -s $s --width_ratio 1 --weight_learning --criterion crossentropy

python train_mnist_slth.py -o adam -w 0.1 -l 0.001 -s $s --width_ratio 1 --weight_learning --criterion mse --num_epochs 100000 --save_every 2000 

# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 1
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 4 --weight_learning
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 4
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 8 --weight_learning
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 8

