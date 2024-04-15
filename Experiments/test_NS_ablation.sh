#!/bin/bash


for data in 1
do
    for r in 1 2 4
    do
        for m in {1..15}
        do
            python Test.py -bench $data -rollout $r -model $m -final -ylim 0,5 -ready
        done
    done
done

for data in 2 3
do
    for r in 1 2 4 8
    do
        for m in {1..15}
        do
            python Test.py -bench $data -rollout $r -model $m -final -ylim 0,30 -ready
        done
    done
done
