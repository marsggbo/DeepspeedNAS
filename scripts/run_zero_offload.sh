#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config ./configs/zero3_offload_config.json \
--use_zero 1 --zero-stage 3 --offload 1 --use_fp16 1 \
$@
