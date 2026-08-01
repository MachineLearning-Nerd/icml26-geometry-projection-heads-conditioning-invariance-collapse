"""Backbone and projection head, matching the paper's Appendix C description.

ResNet-18 adapted to 32x32 CIFAR inputs (3x3 stem, no maxpool) giving a 512-d
representation z; projection head is a 2-layer MLP 512 -> 2048 -> 2048 with
bias-free linear layers, optional BatchNorm, and a configurable activation.
`activation='linear'` drops the nonlinearity entirely (two stacked linear maps),
which is the exact object of Theorem 4.1 part 1.
"""

import torch.nn as nn
import torchvision


class ResNetBackbone(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.output_dim = output_dim

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "swish": nn.SiLU}


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048,
                 activation="relu", use_bn=False):
        super().__init__()
        if activation == "linear":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=False),
                nn.Linear(hidden_dim, output_dim, bias=False),
            )
            return
        layers = [nn.Linear(input_dim, hidden_dim, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(ACTIVATIONS[activation]())
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        if use_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
