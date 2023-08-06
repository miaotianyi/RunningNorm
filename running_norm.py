from typing import Sequence, Union
from collections import abc

import torch
from torch import nn


class RunningNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor
    num_batches_tracked: torch.Tensor

    def __init__(self,
                 kept_axes: Union[int, Sequence[int]],
                 kept_shape: Union[int, Sequence[int]],
                 eps: float = 1e-5,
                 momentum: float = None,
                 device=None, dtype=None) -> None:
        """
        Standardize input tensor to zero mean and unit variance,
        like `StandardScaler` in Scikit-Learn, with additional support for:

        1. Minibatch training and inference.
        2. Arbitrary input tensor dimension and shape.
        3. Arbitrary dimensions to standardize over.
        4. Easy integration into `nn.Sequential` with one line of code.

        - Input: `(*)`, tensor to standardize.
        - Output: `(*)`, standardized tensor with the same shape.

        The formula for the standardized output `z` is
        `z = (x - u) / s`, where `u` is the running mean and
        `s` is the running standard deviation.

        During training, for each batch `x`,
        the per-batch mean is `x.detach().mean(dim=reduce_axes, keepdim=True)`,
        the per-batch variance is `x.detach().var(dim=reduce_axes, keepdim=True)`,
        where `reduce_axes` are all the axes not in `kept_axes`.

        - To standardize each feature in a `[N, C]` tabular dataset,
          we can set `kept_axes=1` and `kept_shape=C`.
        - To standardize each channel in a `[N, C, H, W]` image dataset,
          we can set `kept_axes=1` and `kept_shape=C`.
        - To standardize each pixel in a `[N, C, H, W]` image dataset,
          we can set `kept_axes=[-2, -1]` (or `[2, 3]`) and `kept_shape=[H, W]`.

        The running mean and running variance are updated by
        `running_stat := (1 - alpha) * running_stat + alpha * batch_stat`
        before standardization.

        - If `momentum` is `None`, we use cumulative moving average via
          `alpha = 1 / num_batches_tracked`.
        - If `momentum` is a float between 0 and 1, we use
          exponential moving average via `alpha = momentum`.

        During inference, per-batch statistics are not calculated.

        Note: BatchNorm cannot be directly used as a substitute
        for StandardScaler, because BatchNorm uses gradient-tracking
        per-batch mean and variance for standardization.

        Parameters
        ----------
        kept_axes : int, sequence of int, default: (1,)
            The axes to keep; other axes are reduced for mean and variance.

            Must have the same length as `kept_shape`.

        kept_shape : int, sequence of int
            The length of each kept axis.

            If there are multiple axes to keep, a list of integers must be provided.
            The running mean and variance tensors have shape `kept_shape`.

        eps : float, default: 1e-5
            A value added to the denominator for numerical stability
            in `std = sqrt(running_var + eps)`.

        momentum : float, default: None
            The value used for the `running_mean` and `running_var` updates.

            It's the "weight" placed on the new batch statistics;
            higher momentum (smoothing factor) assigns greater importance
            to current statistics. This is the opposite convention from
            momentum in optimizers. We stick to BatchNorm's convention.

            Default to `None` for cumulative moving average (i.e. simple average).
        """
        super(RunningNorm, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # check kept_axes
        if isinstance(kept_axes, abc.Sequence):  # list of int
            self.kept_axes = tuple(int(s) for s in kept_axes)
        else:   # int
            self.kept_axes = (int(kept_axes), )
        # check kept_shape
        if isinstance(kept_shape, abc.Sequence):
            assert len(kept_shape) == self.ndim,\
                f"kept_shape {kept_shape} and kept_axes {kept_axes} must have the same length"
            self.kept_shape = tuple(int(s) for s in kept_shape)
        else:   # int; broadcast to list if there's more than 1 kept axis
            self.kept_shape = (int(kept_shape),) * self.ndim

        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(*self.kept_shape, **factory_kwargs))
        self.register_buffer('running_var', torch.ones(*self.kept_shape, **factory_kwargs))
        self.register_buffer('num_batches_tracked',
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.reset_running_stats()

    @property
    def ndim(self) -> int:
        """
        Number of dimensions in the running statistics tensors
        """
        return len(self.kept_axes)

    def reset_running_stats(self) -> None:
        """
        Reset running mean to 0, running variance to 1,
        and number of tracked batches to 0.
        """
        # copied from torch BatchNorm
        # running_mean/running_var/num_batches... are registered at runtime
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def _check_input_dim(self, x):
        if not all(x.shape[a] == s for a, s in zip(self.kept_axes, self.kept_shape)):
            raise ValueError(f"expected shape {self.kept_shape} at axes {self.kept_axes}, got input shape {x.shape}")

    def extra_repr(self):
        return f"kept_axes={self.kept_axes}, kept_shape={self.kept_shape}, eps={self.eps}, momentum={self.momentum}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if self.training:
            self.num_batches_tracked.add_(1)  # type: ignore[has-type]
            if self.momentum is None:  # use cumulative moving average
                # alpha: exponential_average_factor, learning rate, "same" as momentum
                alpha = 1.0 / float(self.num_batches_tracked)
            else:  # use exponential moving average
                alpha = self.momentum
        else:
            alpha = 0.0

        # represent kept_axes as non-negative integers,
        # so negative kept_axes can be supported.
        srs_axes = tuple(range(x.ndim)[a] for a in self.kept_axes)  # source axes
        tgt_axes = tuple(range(self.ndim))  # target axes

        if self.training:   # update running statistics
            reduce_axes = tuple(a for a in range(x.ndim) if a not in srs_axes)
            # batch_mean = x.detach().mean(dim=reduce_axes, keepdim=True).movedim(srs_axes, tgt_axes).squeeze()
            # batch_mean_x2 = (x.detach() ** 2).mean(dim=reduce_axes, keepdim=True).movedim(srs_axes, tgt_axes).squeeze()
            # batch_var = batch_mean_x2 - batch_mean ** 2
            batch_var, batch_mean = torch.var_mean(x.detach(), dim=reduce_axes, keepdim=True)
            batch_mean = batch_mean.movedim(srs_axes, tgt_axes).squeeze()
            batch_var = batch_var.movedim(srs_axes, tgt_axes).squeeze()
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var

        # running stats back to x
        new_idx = (...,) + (None,) * (x.ndim - self.ndim)   # insert trivial columns at the end
        mean = self.running_mean[new_idx].movedim(tgt_axes, srs_axes)
        std = torch.sqrt(self.running_var + self.eps)[new_idx].movedim(tgt_axes, srs_axes)
        return (x - mean) / std
