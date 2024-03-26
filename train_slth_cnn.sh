#!/usr/bin/zsh 
s=0
# python train_mnist_slth.py -o adam -w 0.1 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.2 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.3 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.4 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.5 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy

python train_mnist_slth.py -o adam -w 0.9 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy --num_epochs 100000 --save_every 2000
python train_mnist_slth.py -o adam -w 0.8 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy --num_epochs 100000 --save_every 2000
python train_mnist_slth.py -o adam -w 0.7 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy --num_epochs 100000 --save_every 2000
python train_mnist_slth.py -o adam -w 0.6 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy --num_epochs 100000 --save_every 2000


# python train_mnist_slth.py -o adam -w 0.1 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion mse
# python train_mnist_slth.py -o adam -w 0.1 -l 0.001 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.01 -l 0.01 -s $s --width_ratio 1 --weight_learning --model cnn --criterion mse
# python train_mnist_slth.py -o adam -w 0.01 -l 0.01 -s $s --width_ratio 1 --weight_learning --model cnn --criterion crossentropy
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 1 --model cnn
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 4 --weight_learning --model cnn
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 4 --model cnn
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 8 --weight_learning --model cnn
# python train_mnist_slth.py -o adam -w 0.01 -l 0.001 -s $s --width_ratio 8 --model cnn

