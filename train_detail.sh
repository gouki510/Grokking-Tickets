#!/usr/bin/zsh 
s=1

python train.py -o adam -w 1 -l 0.001 -s $s --width 48
python train.py -o adam -w 1 -l 0.001 -s $s --width 128
python train.py -o adam -w 1 -l 0.001 -s $s --width 512
python train.py -o adam -w 1 -l 0.001 -s $s --width 1024
python train.py -o adam -w 1 -l 0.001 -s $s --width 8192
python train.py -o adam -w 1 -l 0.001 -s $s --width 16384
python train.py -o adam -w 1 -l 0.001 -s $s --width 65536