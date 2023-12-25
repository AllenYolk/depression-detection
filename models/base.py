import abc

import torch.nn as nn


class BaseNet(nn.Module, abc.ABC):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def feature_extractor(self, x):
        pass

    @abc.abstractmethod
    def classifier(self, x):
        pass

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


class TMeanNet(BaseNet):

    def __init__(self, hidden_sizes=[256, 128], dropout=0.5):
        super().__init__()
        self.fcs = nn.Sequential()
        last_dim = 161
        for h in hidden_sizes:
            self.fcs.append(nn.Linear(last_dim, h))
            self.fcs.append(nn.ReLU())
            self.fcs.append(nn.Dropout(dropout))
            last_dim = h
        self.output = nn.Linear(last_dim, 1)

    def feature_extractor(self, x):
        return x.mean(dim=1)

    def classifier(self, x):
        return self.output(self.fcs(x))
