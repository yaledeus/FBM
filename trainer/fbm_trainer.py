#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import Trainer
from torch import nn


def disable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def enable_grad(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


class FBMTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def get_scheduler(self, optimizer):
        return None

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        log_type = 'Validation' if val else 'Train'

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            loss, (loss_iff, loss_bound) = self.model.module._train(batch)
        else:
            loss, (loss_iff, loss_bound) = self.model._train(batch)

        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'IFF Loss/{log_type}', loss_iff, batch_idx, val)
        self.log(f'Boundary Loss/{log_type}', loss_bound, batch_idx, val)

        return loss
