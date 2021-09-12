#!/usr/bin/env bash

python3 run.py --dataset mnistRGB --model convnet --epochs 50 --processes 2 -j 3

wait
