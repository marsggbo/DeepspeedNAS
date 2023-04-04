#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config ./configs/ddp_config.json \
$@
