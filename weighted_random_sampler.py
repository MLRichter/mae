from torch.nn import CrossEntropyLoss
from torch.utils.data import Sampler
import torch



class WeightedSubsetRandomSampler(Sampler):
    r"""Samples elements from a given list of indices with given probabilities (weights), with replacement.

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
    """

    def __init__(self, indices, weights, num_samples=0):
        if isinstance(num_samples, bool):
            raise ValueError("num_samples should be a non-negative integeral "
                             "value, but got num_samples={}".format(num_samples))
        self.indices = indices
        weights = [ weights[i] for i in self.indices ]
        self.weights = torch.tensor(weights, dtype=torch.double)
        if num_samples == 0:
            self.num_samples = len(self.weights)
        else:
            self.num_samples = num_samples
        self.replacement = True

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples


import torch
import torchvision.transforms
from torchvision import datasets

def make_weights_for_balanced_classes(images, nclasses):
    n_images = len(images)
    count_per_class = [0] * nclasses
    for _, image_class in images:
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, (image, image_class) in enumerate(images):
        weights[idx] = weight_per_class[image_class]
    return weights


def even(weights):
    weights[:] = 1/len(weights)
    return weights

traindir = "../coco2017/"

dataset_train = datasets.ImageFolder(traindir, transform=torchvision.transforms.Compose(
    [torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
)

# For unbalanced dataset we create a weighted sampler
weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
weights = torch.DoubleTensor(weights)
#weights = even(weights)
#sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

#train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=1000,
#                                           sampler=sampler, num_workers=0, pin_memory=True)

#print(weights)
#for batch, labels in train_loader:
#    print(len(batch), "batch size")
#    class_0 = labels[labels == 0]
#    class_1 = labels[labels == 1]
#    class_2 = labels[labels == 2]
#    classes = [class_0, class_1, class_2]
#    for i, cls in enumerate(classes):
#        print(f"class {i}: {len(cls)} ({round((len(cls) / len(labels))*100, 2)}%)")
#    weights = even(weights)
#    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#    sampler.weights.data = weights.data


#class MyDataset(Dataset):
#        ...
     # return data idx here, so that the corresponding weight can be updated based on the training loss.
#    def __getitem__(self, idx):
#        ...
#        return image, label, idx

import numpy as np
total_epochs = 4
batch_size = 1000

criterion_individual = CrossEntropyLoss()
#sample_weights = np.ones(len(dataset_train))
train_set = dataset_train
for epoch in range(0, total_epochs):
        # tr_indices indexes a subset of train_labels
        tr_indices = np.asarray(list(range(len(dataset_train))))
        train_sampler = WeightedSubsetRandomSampler(tr_indices, weights)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=0)

        for batch_i, (data, labels) in enumerate(train_loader):
            print(len(data), "batch size")
            class_0 = labels[labels == 0]
            class_1 = labels[labels == 1]
            class_2 = labels[labels == 2]
            classes = [class_0, class_1, class_2]
            for i, cls in enumerate(classes):
                print(f"class {i}: {len(cls)} ({round((len(cls) / len(labels))*100, 2)}%)")
              # update sample weights to be the loss, so that harder samples have larger chances to be drawn in the next epoch
            weights[:] = even(weights)[:]
            break