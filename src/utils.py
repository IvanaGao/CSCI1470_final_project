# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import json
import torch
import torch.nn as nn
import functools
import subprocess
import logging
import colorlog
import torch.distributed as dist


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name="imagenet", abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)-4s %(levelname)s%(reset)s | %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white'
        }
    )

    # formatter = logging.Formatter("【%(asctime)s|%(levelname)s|%(filename)s|line:%(lineno)s】 %(message)s")

    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_distributed(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '':   # 'RANK' in os.environ and
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        if args.local_rank == 0:
            print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "23233"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["LOCAL_RANK"] = str(args.local_rank)
        os.environ["RANK"] = str(args.rank)
        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    print(f'  == distributed init (rank {args.rank}) done.')

    setup_for_distributed(args.rank == 0)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_groups_and_set_lr(args, model_without_ddp: nn.Module):

    # by default
    # import pdb;pdb.set_trace()
    param_dicts = []
    # （1）model learning rate
    temp_dict = {"params": [], "lr": args.lr_model}
    for n, p in model_without_ddp.named_parameters():
        if not match_name_keywords(n, args.lr_backbone_names) and p.requires_grad:
            temp_dict['params'].append(p)
    param_dicts.append(temp_dict)
    # （2）backbone learning rate
    temp_dict = {"params": [], "lr": args.lr_backbone}
    for n, p in model_without_ddp.named_parameters():
        if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad:
            temp_dict['params'].append(p)
    param_dicts.append(temp_dict)

    return param_dicts