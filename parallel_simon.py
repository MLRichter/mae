
import numpy as np
import torch


num_examples = 10
num_classes = 1000


eps=1e-5

labels = np.eye(num_classes)[np.random.choice(num_classes, num_examples)].astype(int)

probs = np.random.uniform(size=[num_examples, num_classes])
probs /= np.expand_dims(np.sum(probs, axis=1), axis=1)
probs_np = probs
probs = torch.from_numpy(probs)

z = probs.argmax(dim=-1)
zi = probs_np.argmax(axis=-1)

n = num_examples
n_y = num_classes
n_z = num_classes

prob_joint = (torch.eye(n_y)[labels, :].T @ torch.eye(n_z)[z, :]) / n + eps

prob_marginal = torch.eye(n_z)[z, :].sum(axis=0) / n + eps

NCE = (prob_joint * torch.log(prob_joint / prob_marginal[None, :])).sum()

print(NCE.item())


prob_joint = (np.eye(n_y)[labels, :].T @ np.eye(n_z)[zi, :]) / n + eps

prob_marginal = np.eye(n_z)[zi, :].sum(axis=0) / n + eps

NCE = (prob_joint * np.log(prob_joint / prob_marginal[None, :])).sum()

print(NCE.item())