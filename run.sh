for i  in {0..10}; do
    echo "Running $i
    python3 prune.py -p "$i*0.1" 
done
