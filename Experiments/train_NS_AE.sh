#!/bin/bash

epochs=1

for b in 1 2 3
do
    python TrainAE.py -bench $b -epochs $epochs -embed_dim 128
done
