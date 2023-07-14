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


def adjust_mask_rate(epoch, cosine_epochs, mask_rate, linear_epochs, min_mask):
    """Decay the rate of UNMASKED tokens with half-cycle cosine after warmup"""
    if epoch < linear_epochs:
        mask_rate = min_mask * epoch / linear_epochs
    elif epoch > cosine_epochs:
        return mask_rate
    else:
        epoch = cosine_epochs - (epoch - linear_epochs)
        mask_rate = min_mask + (mask_rate - min_mask) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - linear_epochs) / (cosine_epochs - linear_epochs)))
    return mask_rate


"""def plot_mask_rate():
    actual_epochs = 800
    cosine_epochs = 400
    linear_epochs = 0
    min_mask = 0.0
    mask_rates = []

    for epoch in range(actual_epochs + 1):
        mask_rate = adjust_mask_rate(epoch, cosine_epochs, 0.75, linear_epochs, min_mask)
        mask_rates.append(mask_rate)

    plt.plot(range(actual_epochs + 1), mask_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Mask Rate')
    plt.title('Mask Rate Decay')
    plt.show()

plot_mask_rate()


if __name__ == '__main__':

    plot_mask_rate()
"""
