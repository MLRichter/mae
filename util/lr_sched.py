# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_mask_rate(epoch, epochs, mask_rate, warmup_epochs, min_mask):
    """Decay the rate of UNMASKED tokens with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        mask_rate = mask_rate * epoch / warmup_epochs
    else:
        epoch = epochs - (epoch - warmup_epochs)
        mask_rate = min_mask + (mask_rate - min_mask) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    return mask_rate

