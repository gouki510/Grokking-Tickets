#!/usr/bin/zsh 

python train.py -o adam -w 1 -l 0.01 -s 1 --width 48 --is_symmetric_input
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 48
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 128
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 512
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 1024
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 8192
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 16384
# python train.py -o adam -w 1 -l 0.001 -s 0 --width 65536



