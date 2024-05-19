#!/usr/bin/bash
for i in `seq 300`
do
    CUDA_VISIBLE_DEVICES=2 python prune.py -w 1 -s 0 -e $((i*5)) 
done

