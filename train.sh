#!/bin/bash

cd src

CUDA_VISIBLE_DEVICES=0 python main.py --dataset_type LLVIP --MP 30 --load_data_setting iterative

exit 0