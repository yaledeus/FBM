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


class BBMTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        super().__init__(model, train_loader, valid_loader, config)

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        log_type = 'Validation' if val else 'Train'

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            loss, (loss_eu, loss_aux) = self.model.module._train(batch)
        else:
            loss, (loss_eu, loss_aux) = self.model._train(batch)

        self.log(f'Loss/{log_type}', loss, batch_idx, val)
        self.log(f'Eu Loss/{log_type}', loss_eu, batch_idx, val)
        self.log(f'Aux Loss/{log_type}', loss_aux, batch_idx, val)

        return loss
