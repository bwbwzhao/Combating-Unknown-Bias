import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.identity = nn.Identity()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1)
        feat = x = self.feature(x)
        x = self.identity(x)

        x = self.fc(x)

        if return_feat:
            return x, feat
        else:
            return x