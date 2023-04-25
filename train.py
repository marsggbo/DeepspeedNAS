#!/usr/bin/env python3
# https://github1s.com/microsoft/DeepSpeedExamples/blob/HEAD/pipeline_parallelism/train.py

import argparse
import json
import logging
import os
from datetime import datetime
from time import time
import numpy as np
import traceback

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader, log_dist, logger
from hyperbox.mutator import RandomMutator

from model_zoo import get_model
from datasets import cifar_trainset, FakeDataset
from utils import get_memory_usage, ExpLog


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('-s', '--steps', type=int, default=10, help='quit after this many steps')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--use_ac', type=int, default=0, help='use activation checkpointing') # 1: True 0: False
    parser.add_argument('--use_fp16', type=int, default=0) # 1: True 0: False
    parser.add_argument('--use_pipeline', type=int, default=0) # 1: True 0: False
    parser.add_argument('--num_stages', type=int, default=-1, help='number of stages in pipeline') # -1: auto, equal to #gpus
    parser.add_argument('--use_zero', type=int, default=0) # 1: True 0: False
    parser.add_argument('--zero_stage', type=int, default=3) # [1, 2, 3]
    parser.add_argument('--offload', type=str, default='') # 1: True 0: False
    parser.add_argument('--img_size', type=int, default=224, help='use img_size')
    parser.add_argument('--batch_size', type=int, default=32, help='global batch_size')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--model', type=str, default='vit_b', help='model name')
    parser.add_argument('--debug', type=int, default=0, help='enable debug when set to 1')
    parser.add_argument('--tune_cfg', type=int, default=0, help='enable tuning cfg when set to 1')
    parser.add_argument('--exp_name', type=str, default='') # 1: True 0: False
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args, logger=logger, config_params=None):
    exp_log = ExpLog(args)
    rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    logger.info('Normal or DDP mode')
    deepspeed.runtime.utils.set_random_seed(args.seed)

    net = get_model(args.model, args)
    rm = RandomMutator(net)
    trainset = FakeDataset(args.img_size, 10, args.use_fp16)

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset,
        config_params=config_params)
    if args.use_fp16:
        engine.fp16_enabled()
    if args.use_ac:
        logger.info(f'[rank{rank}] To enable activation checkpointing from {args.deepspeed_config}')
        deepspeed.checkpointing.configure(None, args.deepspeed_config)
        logger.info(f'[rank{rank}] Enable activation checkpointing Done')

    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    gas = engine.gradient_accumulation_steps()

    criterion = torch.nn.CrossEntropyLoss()

    total_steps = args.steps * engine.gradient_accumulation_steps()
    step = 0
    for step in range(total_steps):
        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)
        if args.use_fp16:
            inputs = inputs.half()

        start = time()
        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()
        end = time()
        batch_time = end - start
        BS = inputs.shape[0]
        throughput = BS * world_size / batch_time
        calc = f"{BS} (BS) * {world_size}($gpus) / {batch_time:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage()
        if step>5:
            exp_log.add(batch_time, throughput)
        torch.cuda.synchronize()
        if rank == 0:
            logger.info(f'[rank{rank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{rank}] step{step}: CPU Mem ({cm:.2f}) | GPU Mem-A({ag:.2f}) Max-Mem-A({mag:.2f}) Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if rank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_pipe(args, part='uniform', logger=logger, config_params=None):
    exp_log = ExpLog(args)
    logger.info('Pipeline mode')
    deepspeed.runtime.utils.set_random_seed(args.seed)

    net = get_model(args.model, args)
    rm = RandomMutator(net)
    criterion = torch.nn.CrossEntropyLoss()
    if args.use_ac:
        deepspeed.checkpointing.configure(None, args.deepspeed_config)

    rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    num_stages = args.gpus if args.num_stages == -1 else args.num_stages
    assert num_stages >= 1, 'num_stages must be >= 1 but got {}'.format(num_stages)
    net = PipelineModule(layers=net.join_layers(),
                         loss_fn=criterion,
                         num_stages=args.gpus,
                         partition_method=part,
                         activation_checkpoint_interval=0)

    trainset = FakeDataset(args.img_size, 10, args.use_fp16)

    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset,
        config_params=config_params)
    if args.use_fp16:
        engine.fp16_enabled()
    dataloader = RepeatingLoader(dataloader)
    data_iter = iter(dataloader)

    BS = engine.train_batch_size()
    for step in range(args.steps):
        rm.reset()
        torch.cuda.synchronize()
        start = time()
        loss = engine.train_batch(data_iter)
        end = time()
        batch_time = end - start
        throughput = BS / (end - start)
        calc = f"{BS} (BS) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage()
        if step>5:
            exp_log.add(batch_time, throughput)
        torch.cuda.synchronize()
        if rank == 0:
            logger.info(f'[rank{rank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{rank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A{mag:.2f} Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if rank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def train_zero(args, logger=logger, config_params=None):
    exp_log = ExpLog(args)
    logger.info('Zero mode')
    deepspeed.runtime.utils.set_random_seed(args.seed)

    net = get_model(args.model, args)
    rm = RandomMutator(net)
    if args.use_ac:
        deepspeed.checkpointing.configure(None, args.deepspeed_config)

    trainset = FakeDataset(args.img_size, 10, args.use_fp16)
    criterion = nn.CrossEntropyLoss()

    engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=[p for p in net.parameters() if p.requires_grad],
        training_data=trainset,
        config_params=config_params)
    engine.fp16_enabled()
    engine.tput_timer.monitor_memory = True
    trainloader = RepeatingLoader(trainloader)

    rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    for step, data in enumerate(trainloader):
        if step+1 > args.steps:
            break
        rm.reset()
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(engine.local_rank), data[1].to(engine.local_rank)
        BS = inputs.shape[0]
        if args.use_fp16:
            inputs = inputs.half()
        start = time()
        outputs = engine(inputs)
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()
        end = time()
        batch_time = end - start
        throughput = BS* world_size / (end - start)
        calc = f"{BS} (BS) * {world_size}($gpus) / {end - start:.2f}(time)"
        ag, mag, rg, mrg, cm = get_memory_usage()
        if step>5:
            exp_log.add(batch_time, throughput)
        torch.cuda.synchronize()
        if rank == 0:
            logger.info(f'[rank{rank}] step{step}: throughput is {calc}={throughput:.2f} img/sec loss: {loss}')
            logger.info(f'[rank{rank}] step{step}: CPU Mem {cm:.2f} | GPU Mem-A({ag:.2f}) Max-Mem-A({mag:.2f}) Mem-R({rg:.2f}) Max-Mem-R({mrg:.2f})')
    if rank == 0:
        exp_log.save_as_csv()
        exp_log.stats_and_save()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_exp(args, default_cfg_path='./configs/base_config.json'):
    model = args.model
    gpus = args.gpus
    batch_size = args.batch_size
    img_size = args.img_size
    use_pipeline = args.use_pipeline
    use_zero = args.use_zero
    use_fp16 = args.use_fp16
    use_ac = args.use_ac
    gpus = args.gpus
    debug = args.debug
    exp_name = args.exp_name
    assert not (use_pipeline and use_zero), 'Cannot use both pipeline and zero'
    set_seed(args.seed)

    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    name = f'{model}_ds_gpu{gpus}_{batch_size}x{img_size}x{img_size}'
    if use_zero:
        name += f'_zero'
        if args.offload in ['cpu', 'nvme']:
            name += f'_offload({args.offload})'
    if use_pipeline:
        name += f'_pipeline'
    if use_fp16:
        name += f'_fp16'
    if use_ac:
        name += '_ac'
    if exp_name not in ['', 'null']:
        name += f'_{exp_name}'
    root_dir = f"./logs/{name.replace('_ds', '/ds')}/{date_of_run}"
    grank = int(os.environ['RANK'])
    lrank = int(os.environ['LOCAL_RANK'])
    root_dir = f'{root_dir}/rank_{grank}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    global logger
    file_handler = logging.FileHandler(f'{root_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f'exp_name: {name}')
    args.root_dir = root_dir
    
    # parse config params
    # if args.deepspeed_config is None:
        # filepath = args.deepspeed_config
        # delattr(args, 'deepspeed_config')
    with open(default_cfg_path, 'r') as f:
        config_params = json.load(f)
    config_params['train_micro_batch_size_per_gpu'] = args.batch_size
    config_params['deepspeed']['num_gpus'] = args.gpus
    if use_fp16:
        config_params['fp16']['enabled'] = True
    if use_ac:
        config_params['activation_checkpointing']['partition_activations'] = True
    if use_pipeline:
        config_params['zero_optimization']['stage'] = 0
    if use_zero:
            config_params['zero_optimization']['stage'] = args.zero_stage
            if args.offload in ['cpu', 'nvme']:
                config_params['zero_optimization'].update(
                    {        
                        "offload_optimizer": {
                            "device": f"{args.offload}",
                            "nvme_path": "./local_nvme",
                            "pin_memory": True,
                            "buffer_count": 4,
                            "fast_init": False
                        },
                        "offload_param": {
                            "device": f"{args.offload}",
                            "nvme_path": "./local_nvme",
                            "pin_memory": True,
                            "buffer_count": 5,
                            "buffer_size": 1e8,
                            "max_in_cpu": 1e9
                        }
                    })
    if args.tune_cfg:
        config_params["autotuning"]["enabled"] = True
        config_params["train_batch_size"] = 'auto'
        config_params["gradient_accumulation_steps"] = 'auto'

    config_params_file = f'{root_dir}/config.json'
    args_file = f'{root_dir}/args.json'
    args.deepspeed_config = config_params_file
    with open(config_params_file, 'w') as f:
        json.dump(config_params, f, indent=4)
    with open(args_file, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    logger.info(f'args: {args.__dict__}')
    return logger, config_params


if __name__ == '__main__':
    args = get_args()
    if args.debug:
        from ipdb import set_trace
        set_trace()
    logger, config_params = init_exp(args)
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    try:
        if args.use_pipeline != 0:
            train_pipe(args, logger=logger, config_params=config_params)
        elif args.use_zero != 0:
            train_zero(args, logger=logger, config_params=config_params)
        else:
            train_base(args, logger=logger, config_params=config_params)
    except BaseException as e:
        logger.info(traceback.format_exc())
