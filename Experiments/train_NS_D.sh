#!/bin/bash

epochs=100

for b in 1 2 3
do
    for rollout in 1 2 4
    do
        python Train.py -bench $b -dts 1,2,4,8 -rollout $rollout -attn 5 -pos 4 -epochs $epochs
        python Train.py -bench $b -dts 1,2,4,8 -rollout $rollout -attn 5 -pos 4 -epochs $epochs
        python Train.py -bench $b -dts 1,2,4,8 -rollout $rollout -attn 5 -pos 4 -epochs $epochs
    done
done



