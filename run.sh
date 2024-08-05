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

