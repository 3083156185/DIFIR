import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

from .utils.img import flip, shuffleLR
from .models.FastPose import FastPose

import sys

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2



class InferenNet_fastRes50(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.pyranet = FastPose('resnet50', 17).cpu()
        print('Loading pose model from {}'.format(model_path))
        self.pyranet.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.pyranet.eval()

    def forward(self, x):
        out = self.pyranet(x)

        return out
