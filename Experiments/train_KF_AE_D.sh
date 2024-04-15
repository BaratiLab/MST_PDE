#!/bin/bash

epochs=100

python TrainAE.py -bench 7 -epochs 100 -embed_dim 128 -down 4
python Train.py -epochs 100 -bench 7 -dts 1 -rollout 8 -attn 5 -pos 4 -down 4 -embed_dim 128 -heads 8,8,8,8,8,8,8,8
