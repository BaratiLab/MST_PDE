#!/bin/bash

epochs=1

for d in 1 2 3
do
    for r in 1 2 4 8
    do
        for i in 1 2 3
        do
            # M0
            python Train.py -epochs $epochs -bench $d -dts 1,2,4,8 -rollout $r -attn 0
        done
        for i in 1 2 3
        do
            #M1
            python Train.py -epochs $epochs -bench $d -dts 1,2,4,8 -rollout $r -attn 1 -pos 0
        done
        for i in 1 2 3
        do
            # M2
            python Train.py -epochs $epochs -bench $d -dts 1,2,4,8 -rollout $r -attn 1 -pos 1
        done
        for i in 1 2 3
        do
            # M3
            python Train.py -epochs $epochs -bench $d -dts 1,2,4,8 -rollout $r -attn 2 -pos 4
        done
        for i in 1 2 3
        do
            # M4
            python Train.py -epochs $epochs -bench $d -dts 1,2,4,8 -rollout $r -attn 5 -pos 4
        done
    done
done