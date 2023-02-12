import torch.nn as nn
from module.resnet import resnet20
from module.mlp import MLP
from module.resnet_B import resnet18, resnet50
import torch
import torch.nn.functional as F
import io
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F
import math
import torch.optim as optim


def get_model(model_tag, num_classes, ss=False):
    if model_tag == "ResNet20":
        model = resnet20(num_classes, ss=ss)
        return model

    elif model_tag == "ResNet18":
        model = resnet18(pretrained=True, ss=ss)
        model.fc = nn.Linear(512, num_classes)
        return model

    elif model_tag == "MLP":
        model = MLP(num_classes)
        return model
        
    else:
        raise NotImplementedError
