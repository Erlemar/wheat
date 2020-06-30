import torch
from torch import nn

import torch.functional as F
from torch.nn.parameter import Parameter

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps=self.eps)"
        # return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
        #     self.eps) + ')'

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


import torch
from torch import nn


class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.

    # https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/modules/rms_norm.html#RMSNorm
    """

    def __init__(
        self, dimension: int, epsilon: float = 1e-8, is_bias: bool = False
    ):
        """
        Args:
            dimension (int): the dimension of the layer output to normalize
            epsilon (float): an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias (bool): a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_std = torch.sqrt(torch.mean(x ** 2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if self.is_bias:
            return self.scale * x_norm + self.bias
        return self.scale * x_norm
