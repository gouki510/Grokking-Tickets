#!/usr/bin/zsh 
array=("snip"  "rand" "mag" "grasp" "synflow")
array2=("add", "subtract", "x2xyy2","rand")
for j in {0..0} 
    do 
    for i in {1..29}
        do
            python prune.py  -s $j   -p $((0.1+i*0.01)) 
        done
    done

