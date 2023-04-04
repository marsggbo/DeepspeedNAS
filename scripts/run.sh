#!/bin/bash

deepspeed train.py --deepspeed --deepspeed_config ./configs/base_config.json $@
