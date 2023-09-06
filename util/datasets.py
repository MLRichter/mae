# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import tarfile
from os.path import isfile

import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from fast_imagenet import ImageNetDatasetH5
from tarbal_parser import ParserImageTar

imnet21k_cache = {}


def build_dataset(is_train, args, transforms=None):
    transform = transforms if transforms is not None else build_transform(is_train, args)
    if args.data_path.endswith("hdf5"):
        print("Detected file instead of folder, assuming hdf5")
        dataset = ImageNetDatasetH5(args.data_path, split='train' if is_train else 'val', transform=transform)
    elif args.data_path.endswith("cifar100"):
        print('Enabled Cifar10 training', args.data_path)
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=False)
    elif args.data_path.endswith("cifar10"):
        print('Enabled Cifar10 training', args.data_path)
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=False)
    elif "food-101" in args.data_path:
        from dataset import ImageDataset
        print("Training on FOOD101")

        if "train" not in imnet21k_cache:
            with tarfile.open(args.data_path) as tf:  # cannot keep this open across processes, reopen later
                train = ParserImageTar(args.data_path, tf=tf, subset="train")
                val = ParserImageTar(args.data_path, tf=tf, subset="test")
                imnet21k_cache["train"] = train
                imnet21k_cache["val"] = val
        dataset = ImageDataset(root=args.data_path,
                               reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        args.nb_classes = len(imnet21k_cache["val"].class_to_idx)
    elif "eurosat" in args.data_path:
        from dataset import ImageDataset
        print("Training on EuroSat")

        if "train" not in imnet21k_cache:
            with tarfile.open(args.data_path) as tf:  # cannot keep this open across processes, reopen later
                train = ParserImageTar(args.data_path, tf=tf, subset="train")
                val = ParserImageTar(args.data_path, tf=tf, subset="val")
                imnet21k_cache["train"] = train
                imnet21k_cache["val"] = val

        dataset = ImageDataset(root=args.data_path,
                               reader=imnet21k_cache["train"] if is_train else imnet21k_cache["val"],
                               transform=transform)
        args.nb_classes = len(imnet21k_cache["val"].class_to_idx)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
