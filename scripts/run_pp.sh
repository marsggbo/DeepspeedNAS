#!/bin/bash
echo "所有传入的参数是：$@"
deepspeed  --master_port 1234 train.py \
--deepspeed_config=./configs/pp_config.json -use_pipeline 1 \
$@
