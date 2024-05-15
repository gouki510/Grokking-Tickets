s=0
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/400.pth --pruning_rate 0.4 --exp_tag edge_pop_w_decay --weight_learning
CUDA_VISIBLE_DEVICES=1  python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag edge_pop_wo_decay --weight_learning --double_reg
s=1
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag edge_pop_w_decay --weight_learning
CUDA_VISIBLE_DEVICES=1 python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag edge_pop_wo_decay --weight_learning --double_reg
s=2
# python train_slth.py -o adam -w 1 -l 0.001 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag edge_pop_w_decay --weight_learning
CUDA_VISIBLE_DEVICES=1 python train_slth.py -o adam -w 1 -l 0.01 -s $s --width_ratio 1 --weight_init_path /workspace/Grokking-Tickets/0120/slth/WIDTHRATIO_1_WEIGHTLR_True/1000.pth --pruning_rate 0.4 --exp_tag edge_pop_wo_decay --weight_learning --double_reg