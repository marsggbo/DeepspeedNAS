import csv
import os
import gc
import pandas as pd

import psutil
from deepspeed.accelerator import get_accelerator

torch_memory_reserved = get_accelerator().memory_reserved
torch_max_memory_reserved = get_accelerator().max_memory_reserved

def get_memory_usage(to_round=True):
    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    MA = get_accelerator().memory_allocated() / (1024**3) # real memory allocated
    Max_MA = get_accelerator().max_memory_allocated() / (1024**3) # peak memory allocated
    CA = torch_memory_reserved() / (1024**3) # memory reserved by torch, >= MA
    Max_CA = torch_max_memory_reserved() / (1024**3) # peak memory reserved by torch, >= Max_MA

    vm_stats = psutil.virtual_memory()
    used_GB = (vm_stats.total - vm_stats.available) / (1024**3)
    # get the peak memory to report correct data, so reset the counter for the next call
    get_accelerator().reset_peak_memory_stats()
    if to_round:
        return round(MA, 2), round(Max_MA, 2), round(CA, 2), round(Max_CA, 2), round(used_GB, 2)
    return MA, Max_MA, CA, Max_CA, used_GB


class ExpLog:
    def __init__(self, args):
        self.args = args
        self.root_dir = self.args.root_dir
        self.data = []
        self.params = {}
        self.init()

    def init(self):
        headers = ['model', 'gpus', 'batch_size', 'img_size', 'use_zero', 'zero_stage', 'use_pipeline', 'num_stages', 'use_fp16', 'use_ac', 'offload', 'seed']
        for h in headers:
            self.params[h] = getattr(self.args, h)

    def add(self, bt, tp):
        ag, mag, rg, mrg, cm = get_memory_usage()
        self.data.append({
            'Batch Time': bt,
            'Throughput': tp,
            'Allocated GPU Mem': ag,
            'Max Allocated GPU Mem': mag,
            'Reserved GPU Mem': rg,
            'Max Reserved GPU Mem': mrg,
            'Used CPU Mem': cm
        })

    def save_as_csv(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.root_dir, 'exp_log.csv')
        df_data = pd.DataFrame(self.data)
        df_params = pd.DataFrame(self.params, index=[0])
        df_params = pd.concat([df_params]*df_data.shape[0], ignore_index=True)
        df = pd.concat([df_params, df_data], axis=1)
        df.to_csv(file_path, index=False)

    def stats_and_save(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.root_dir, 'stat_log.csv')
        df_data = pd.DataFrame(self.data)
        df_params = pd.DataFrame(self.params, index=[0])
        df_stats = {}
        for item in ['Batch Time', 'Throughput', 'Allocated GPU Mem', 'Max Allocated GPU Mem', 'Reserved GPU Mem', 'Max Reserved GPU Mem', 'Used CPU Mem']:
            df_stats[item] = [df_data[item].mean(), df_data[item].var(), df_data[item].max()]
        df_stats = pd.DataFrame(df_stats)
        df_params = pd.concat([df_params]*df_stats.shape[0], ignore_index=True)
        df = pd.concat([df_params, df_stats], axis=1)
        df.to_csv(file_path, index=False)
