#!/bin/bash


# -samples -curves -final -ylim 0,5

# python Test.py -bench $data -rollout $r -model $m -samples 5 -use_dts 1,2 -samples -curves -final -ylim 0,5

data=1
for r in 1 2 4
do
    for m in {13..15}
    do
        for use_dts in 1 1,2 1,2,4 1,2,4,8
        do
            python Test.py -bench $data -rollout $r -model $m -samples 5 -anim -use_dts $use_dts
        done
    done
done