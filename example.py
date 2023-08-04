import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from sklearn.preprocessing import StandardScaler


from running_norm import RunningNorm


def example_1():
    # Tabular dataset with input shape [n_samples, n_features]
    # and output shape [n_samples, n_targets]
    n_samples, n_features, n_targets = 1000, 42, 11
    x = torch.randn(n_samples, n_features)

    # Each feature has parameters mean != 0 and standard deviation != 1
    mu = torch.randn(n_features) * 100
    sigma = torch.exp(torch.randn(n_features))
    x = x * sigma + mu
    # the statistics of mean and variance
    # might not be the same as the parameters
    mean = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False)

    # target y is a linear function of x
    weight = torch.randn(n_features, n_targets) * 5
    bias = torch.randn(n_targets)
    y = torch.matmul(x, weight) + bias

    # batch fitting
    rn = RunningNorm(kept_axes=-1, kept_shape=n_features)
    rn(x)
    print(rn.running_mean - mean)
    print(rn.running_var - var)

    print(rn(x))
    print(rn(x))
    print(rn(x).mean(dim=0))
    print(rn(x).var(dim=0))
    print(np.std(rn(x).numpy(), axis=0))
    ss = StandardScaler()
    print(ss.fit_transform(x.numpy()).std(axis=0))


if __name__ == '__main__':
    example_1()
