#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1234 my_sft.py > SFT_NLFT800.txt 2>&1