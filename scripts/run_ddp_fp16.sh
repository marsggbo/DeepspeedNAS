#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config ./configs/ddp_fp16_config.json \
--use_fp16 1 \
$@
