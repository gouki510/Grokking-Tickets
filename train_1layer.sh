#!/usr/bin/zsh 

python train.py -o adam -w 1 -l 0.01 -s 0 --width 48 --is_1layer --is_symmetric_input
python train.py -o adam -w 1 -l 0.01 -s 0 --width 48 --is_1layer
python train.py -o adam -w 1 -l 0.01 -s 0 --width 48 --is_symmetric_input
python train.py -o adam -w 1 -l 0.01 -s 0 --width 48




