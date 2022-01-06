__all__ = [
    "Model",
]

import logging
from typing import NoReturn, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F  # noqa

from .types import PoisonLearner

NUM_CLASSES = 10
USE_FULL_SIZE = False
USE_DROPOUT = False


class Model(PoisonLearner):
    def __init__(self):
        super(Model, self).__init__(n_classes=NUM_CLASSES)

        div = 1 if USE_FULL_SIZE else 2
        logging.info(f"Using {'full' if USE_FULL_SIZE else 'small'} speech model with div={div}")
        # Note: noqa below due to type resolve errors when params use ints instead of tuples
        self.conv1 = nn.Conv2d(3, 96 // div, kernel_size=11, stride=4)  # noqa
        self.bn1 = nn.BatchNorm2d(num_features=96 // div)
        self.conv2 = nn.Conv2d(96 // div, 256 // div, kernel_size=5,  # noqa
                               stride=1, padding=2, groups=2)  # noqa
        self.bn2 = nn.BatchNorm2d(num_features=256 // div)
        self.conv3 = nn.Conv2d(256 // div, 384 // div, kernel_size=3, stride=1, padding=1)  # noqa
        self.bn3 = nn.BatchNorm2d(num_features=384 // div)
        self.conv4 = nn.Conv2d(384 // div, 384 // div, kernel_size=3,  # noqa
                               stride=1, padding=1, groups=2)  # noqa
        self.bn4 = nn.BatchNorm2d(num_features=384 // div)
        self.conv5 = nn.Conv2d(384 // div, 256 // div, kernel_size=3,  # noqa
                               stride=1, padding=1, groups=2)  # noqa
        self.bn5 = nn.BatchNorm2d(num_features=256 // div)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.drop6 = nn.Dropout(p=0.5)
        self.drop7 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32*4*4, 512)
        # self.output = nn.Linear(512, 1)

        # Encapsulate the
        self.all_conv = nn.Sequential(self.conv1, self.pool1, self.bn1,
                                      self.conv2, self.pool2, self.bn2,
                                      self.conv3, nn.ReLU(), self.bn3,
                                      self.conv4, nn.ReLU(), self.bn4,
                                      self.conv5, nn.ReLU(), self.pool5, self.bn5,
                                      )

    def build_fc(self, x: Tensor, hidden_dim: Optional[int] = None) -> NoReturn:
        r""" Overrides the default hidden layer construction script """
        assert hidden_dim is None, "Hidden dim not supported for speech NN"

        self.eval()
        with torch.no_grad():
            x = self.forward(x=x, penu=True)

        # Hidden linear layers
        in_dim = x.shape[1]
        flds = zip([256, 128], [self.drop6, self.drop7])
        for i, (out_dim, dropout_layer) in enumerate(flds):
            block = nn.Sequential(nn.Linear(in_dim, out_dim),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(out_dim))
            if USE_DROPOUT:
                block.add_module("Dropout{i}", dropout_layer)
            self.fc_first.add_module(f"MLP{i + 1}", block)
            in_dim = out_dim

        # Output layer
        self.linear = nn.Linear(in_dim, self._n_classes)

    def conv_only(self) -> nn.Sequential:
        return nn.Sequential(self.conv1,
                             self.pool1,
                             self.bn1,
                             self.conv2,
                             self.pool2,
                             self.bn2,
                             self.conv3,
                             self.bn3,
                             self.conv4,
                             self.bn4,
                             self.conv5,
                             self.pool5,
                             self.bn5,
                             )

    def forward(self, x: Tensor, penu: bool = False, block: bool = False) -> Tensor:
        assert not block, "Block mode not currently supported"

        out = x
        out = self.conv1(out)
        out = self.pool1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = F.relu(out)
        out = self.bn3(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.bn4(out)

        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool5(out)
        out = self.bn5(out)

        out = self.flatten(out)
        out = self.fc_first(out)

        if penu:
            return out
        out = self.linear(out)
        return out

    def loss(self, pred, label):
        if self.gpu:
            label = label.cuda()
        label = label.float()
        return F.binary_cross_entropy_with_logits(pred, label)
