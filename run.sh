# #!/bin/bash
for i in {0..20}
do
    P=$(echo "$i * 0.05" | bc)
    # python prune.py -p $P -wd 4.8
    # python prune.py -p $P -wd 4.9
    python prune.py -p $P -wd 1
    # python prune.py -p $P -wd 5.1
    # python prune.py -p $P -wd 5.2
    # python prune.py -p $P -wd 5.3
    # python prune.py -p $P -wd 5.4
    # python prune.py -p $P -wd 5.5
done

# CUDA_VISIBLE_DEVICES=1 python train_batch.py -s 0 -o adam -w 1 -l 0.01 -m 0.9 -p 2 -lp 1 -lp_alpha 0.01 -sparse_k 0.01 -model mlp -is_div True -is_symmetric_input True -save_models True -save_every 100