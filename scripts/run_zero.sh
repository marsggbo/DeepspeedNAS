#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config ./configs/zero3_config.json \
--use_zero  1 --zero-stage 3 --fp16 1 \
$@
