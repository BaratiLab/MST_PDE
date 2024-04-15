#!/bin/bash

# dataset
d=7
# rollout
r=8
# model index
m=1

for seed in {0..20}
do
    for start_end in 0,40 50,90 100,140 150,190
    do
        python Test.py -bench $d -dts 1 -rollout $r -model $m -use_dts 1 -ylim 0,120 -samples 1 -start_end $start_end -show_error -cols 5 -seed $seed
    done
done