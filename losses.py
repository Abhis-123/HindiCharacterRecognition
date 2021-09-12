
import torch 
from torch import nn 

def Loss(prediction,label):
    if not torch.is_floating_point(label):
        label = label/1.0

    criterion = nn.BCELoss()
    return criterion(prediction,label)
