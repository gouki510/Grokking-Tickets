#!/usr/bin/zsh 
s=0

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width 48 --weight_learning True
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width 128 --weight_learning True
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4 --weight_init_path /home/0120/slth/WIDTHRATIO_4_WEIGHTLR_True/400.pth

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 --weight_init_path /home/0120/slth/WIDTHRATIO_8_WEIGHTLR_True/400.pth

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 16 --weight_init_path /home/0120/slth/WIDTHRATIO_16_WEIGHTLR_True/400.pth

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 32 --weight_init_path /home/0120/slth/WIDTHRATIO_32_WEIGHTLR_True/400.pth

# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 64 --weight_init_path /home/0120/slth/WIDTHRATIO_64_WEIGHTLR_True/400.pth


# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.1
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.2
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.3
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag 0201
# python train_slth.py -o adam -w 0 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag 0201
# python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag 0201
# python train_slth.py -o adam -w 2 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.5 --exp_tag 0201
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.5
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.6
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.7
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.8
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /home/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.9

# dir_path="/home/0120/slth/memo_detail/WIDTHRATIO_1_WEIGHTLR_True"
# dirs=`find $dir_path -maxdepth 1  -name *.pth`
# echo $dirs
# for dir in $dirs;
# do
#     python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path $dir --exp_tag memo_detail_rewind
# done

# dir_path="/home/0120/slth/memo_detail/WIDTHRATIO_4_WEIGHTLR_True"
# dirs=`find $dir_path -maxdepth 0 -type f -name *.pth`

# for dir in $dirs;
# do
#     python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 4 --weight_init_path $dir --exp_tag memo_detail_rewind
# done

# dir_path="/home/0120/slth/memo_detail/WIDTHRATIO_8_WEIGHTLR_True"
# dirs=`find $dir_path -maxdepth 0 -type f -name *.pth`

# for dir in $dirs;
# do
#     python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 8 --weight_init_path $dir --exp_tag memo_detail_rewind
# done

# dir_path="/home/0120/slth/memo_detail/WIDTHRATIO_16_WEIGHTLR_True"
# dirs=`find $dir_path -maxdepth 0 -type f -name *.pth`


# for dir in $dirs;
# do
#     python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 16 --weight_init_path $dir
# done
