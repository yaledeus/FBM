#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED

### set backend == "pytorch"
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

setup_seed(SEED)

########### Import your packages below ##########
from trainer import TrainConfig
from config import train_config


def main(args):
    ########### load your train / valid set ###########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1

    from data import collate_fn
    if args.dataset == 'pep':
        from data import PepDataset
        train_set = PepDataset(args.train_set)
        valid_set = PepDataset(args.valid_set)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle) \
            if len(args.gpus) > 1 else None
        train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=(args.shuffle and train_sampler is None),
                                  sampler=train_sampler,
                                  collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    elif args.dataset == 'mpep':
        from data import MultiPepDataset, MultiPepBatchSampler, MultiPepDistributedBatchSampler
        train_set = MultiPepDataset(args.train_set, mode='train')
        valid_set = MultiPepDataset(args.train_set, mode='valid')
        train_sampler = MultiPepDistributedBatchSampler(train_set, args.batch_size, shuffle=args.shuffle) \
            if len(args.gpus) > 1 else MultiPepBatchSampler(train_set, args.batch_size, shuffle=args.shuffle)
        valid_sampler = MultiPepBatchSampler(valid_set, args.batch_size, shuffle=False)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers,
                                  collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_sampler=valid_sampler, num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    else:
        print_log(f'[!] Dataset type {args.dataset} not implemented', level='ERROR')
        return

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(args.save_dir, args.lr, args.max_epoch,
                         patience=args.patience,
                         grad_clip=args.grad_clip,
                         save_topk=args.save_topk)

    if args.model_type == 'SFM':
        from trainer import SFMTrainer as Trainer
        from module import GeomSFM
        if not args.ckpt:
            model = GeomSFM(args.hidden_dim, args.rbf_dim, args.heads, args.layers, cutoff=args.cutoff, s_eu=args.s_eu)
        else:
            model = torch.load(args.ckpt, map_location='cpu')
    elif args.model_type == 'FSFM':
        from trainer import FSFMTrainer as Trainer
        from module import GeomFSFM
        if not args.ckpt:
            baseline = torch.load(args.baseline, map_location='cpu')
            model = GeomFSFM(baseline, args.hidden_dim, args.rbf_dim, args.heads, args.layers, cutoff=args.cutoff)
        else:
            model = torch.load(args.ckpt, map_location='cpu')
    else:
        print_log(f'[!] Model type {args.model_type} not implemented', level='ERROR')
        return

    torch.set_default_dtype(torch.float32)

    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = train_config()
    main(args)
