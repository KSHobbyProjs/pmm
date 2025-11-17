#/usr/bin/bash

SEEDS=(0 680 530 504 346 427 536 929 126 136 995)

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    echo "Working through seed ${seed}"
    ./run_pmm.py sample.dat -k 2 -L='-2.0,2.0:50' --config-file config.txt -o "predict${i}.dat" -c dim=2,num_primary=2,seed="$seed" 
done

