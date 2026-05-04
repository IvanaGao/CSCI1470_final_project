import os
import sys
import json
import numpy
import random
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.utils import setup_logger, setup_distributed, save_on_master, get_param_groups_and_set_lr
from src.model import ADRsModel
from src.dataset import ADRsDataset, ADRsDatasetSgTp, ADRsDatasetSgTpAndMultiDrug, collate_fn
from engine import train_one_epoch, evaluation


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# Current script file
script_path = os.path.abspath(__file__)
# Project directory
project_path = script_path.rsplit('/', 1)[0] + '/'
# Configure environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer ADRs', add_help=False)
    # training parameters
    parser.add_argument(
        '--seed',
        # default=None,
        default=33091922,
        type=int
    )
    parser.add_argument('--print_fre', default=10 , type=int)
    parser.add_argument('--lr_model', default=1e-4 , type=float)
    parser.add_argument('--lr_backbone', default=1e-5 , type=float)
    parser.add_argument('--lr_backbone_names', default=['text_encoder'], type=list)
    parser.add_argument('--weight_decay', default=1e-5 , type=float)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--epochs', default=64, type=int)
    parser.add_argument('--max_labels', default=1024, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=0, type=int, help='number of distributed processes')
    parser.add_argument(
        '--eval',
        default=True,
        # default=False,
        type=str
    )
    parser.add_argument(
        '--output_dir',
        default=os.path.join(project_path, os.path.join(project_path, 'log/0811-7/')),
        help='path where to save, empty for no saving'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='resume from checkpoint'
    )
    parser.add_argument('--pretrain_model_path', type=str, default=None, help='load from other checkpoint')
    parser.add_argument(
        '--bert_path',
        type=str,
        default='/mnt/home/xxx/models/bert/bert_uncased_L4_H512_A8',
        help=''
    )
    parser.add_argument('--thr', default=0.3, type=float, help='device to use for training / testing')


    # Multi-node multi-GPU training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')

    # ADR scenario-specific parameters
    parser.add_argument("--only_single_reaction", default=True, type=bool, help='')
    parser.add_argument("--shuffle", default=True, type=bool, help='dataset shuffle and then split for train and val')
    parser.add_argument("--use_class_weights", default=False, type=bool, help='using fucos loss by class weights')
    parser.add_argument("--use_unii_desc", default=False, type=bool, help='using unii desc info')
    parser.add_argument("--use_drug_name", default=True, type=bool, help='using unii desc info')
    # Subgroup parameters
    parser.add_argument("--use_gender", default=True, type=bool, help='using gender info')
    parser.add_argument("--use_age", default=False, type=bool, help='using age info')
    parser.add_argument("--use_weight", default=False, type=bool, help='using weight info')

    args = parser.parse_args()  #Additional namespaces can be integrated into the model

    # Calculate learning rate decay epochs
    parser.add_argument("--multi_step_lr", default=True, type=bool, help='')
    lr_drop_list = [60, 62]
    parser.add_argument("--lr_drop_list", default=lr_drop_list, type=list, help='')
    args = parser.parse_args()

    return args


def seed_all_rng(seed=None, logger=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    from datetime import datetime
    if seed is None:
        seed = (os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big"))
        if logger is None:
            print("Using a generated random seed {}".format(seed))
        else:
            logger.info("Using a generated random seed {}".format(seed))

    numpy.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


if __name__ == "__main__":

    args = get_args_parser()
    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Identify distributed configuration parameters
    setup_distributed(args)

    # Logging module
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'log.txt' if not args.eval else 'log_eval.txt'),
        distributed_rank=args.rank,
        color=False,
        name="adrs"
    )
    logger.info('launching logger ...')
    logger.info('all args = {}'.format(json.dumps(vars(args), indent=4)))

    # set seed
    seed = seed_all_rng(seed=args.seed, logger=logger)

    if not args.eval:
        if args.only_single_reaction:
            dataset_train = ADRsDatasetSgTp('./output/adverse_event_2024.json', logger=logger, args=args)
        else:
            dataset_train = ADRsDataset('./output/adverse_event_2024.json', logger=logger)

    assert args.only_single_reaction is True
    if args.only_single_reaction:
        dataset_val = ADRsDatasetSgTp('./output/adverse_event_2024.json', is_train=False, logger=logger, args=args)
        dataset_val_md = ADRsDatasetSgTpAndMultiDrug(dataset=dataset_val,logger=logger)
    else:
        dataset_val = ADRsDataset('./output/adverse_event_2024.json', is_train=False, logger=logger)
        dataset_val_md = ADRsDatasetSgTpAndMultiDrug(dataset=dataset_val, logger=logger)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_val_md = DistributedSampler(dataset_val_md, shuffle=False)
        if not args.eval:
            sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_val_md = torch.utils.data.SequentialSampler(dataset_val_md)
        if not args.eval:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if not args.eval:
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, batch_size=args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset=dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, sampler=sampler_val, drop_last=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)
    data_loader_val_md = DataLoader(dataset=dataset_val_md, batch_size=args.batch_size, sampler=sampler_val_md, drop_last=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    logger.info('building model ...')
    model = ADRsModel(args=args, class_weights=dataset_val.class_weights)
    model.to(torch.device(args.device))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("learning params size = {}M".format(n_parameters / 1000000))

    # Configure distributed training with DDP wrapping
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params
        )
        model._set_static_graph()
        model_without_ddp = model.module

    param_groups_dicts = get_param_groups_and_set_lr(args=args, model_without_ddp=model_without_ddp)

    # Build the optimizer and set the learning rate decay strategy
    optimizer = torch.optim.AdamW(param_groups_dicts, lr=args.lr_model, weight_decay=args.weight_decay)
    if args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
        args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        # logger.info('load model checkpoint from {}'.format(args.resume))

    if args.resume:
        logger.info('load model checkpoint from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        # Load model weights and resume optimizer and other states from checkpoints
        _load_state_output = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1

    # If no checkpoint is available, initialize the model with weights from pretrain_model_path
    if (not args.resume) and args.pretrain_model_path:
        logger.info('load model from args.pretrain_model_path = {}'.format(args.pretrain_model_path))
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
        _load_state_output = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        logger.info('_load_state_output = {}'.format(str(_load_state_output)))

    # evaluation
    result = evaluation(model, data_loader_val, args=args, logger=logger)
    result_md = evaluation(model, data_loader_val_md, args=args, logger=logger)
    if not args.eval:
        logger.info('start training ...')
        for epoch in range(args.start_epoch, args.epochs):
            train_one_epoch(
                args=args,
                model=model,
                data_loader_train=data_loader_train,
                optimizer=optimizer,
                epoch=epoch,
                logger=logger,
            )

            result = evaluation(model, data_loader_val, args=args, logger=logger)
            result_md = evaluation(model, data_loader_val_md, args=args, logger=logger)

            checkpoint_paths = [
                os.path.join(args.output_dir, 'checkpoint.pth'),
                os.path.join(args.output_dir, f'checkpoint{epoch}.pth'),
            ]

            lr_scheduler.step()

            for checkpoint_path in checkpoint_paths:
                weights = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler,
                    'epoch': epoch,
                    'args': args,
                }
                logger.info('saving {}'.format(checkpoint_path))
                save_on_master(weights, checkpoint_path)
