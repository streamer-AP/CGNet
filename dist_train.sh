#!/bin/bash
NUM_PROC=$1
shift
torchrun --master_port 29505 --nproc_per_node=$NUM_PROC train.py 