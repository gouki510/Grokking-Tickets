#!/usr/bin/bash
for i in `seq 15`
do
    echo $i /15
    CUDA_VISIBLE_DEVICES=2 python prune.py -w 1 -s 0 -e $((i*2000)) --root 20240517/ticket_1 --pre_root /workspace/Grokking-Tickets/20240515/detail/seed1
done

