import torch
import torch.nn.functional as F


def bce_loss(prediction, target, *args, **kwargs) -> torch.Tensor:
    kw = dict()
    device = kwargs["device"]
    if "weight" in kwargs:
        kw = dict(weight=kwargs["weight"].to(device))
    return F.binary_cross_entropy_with_logits(prediction.float().to(device), target.float().to(device), **kw)


def ce_loss(prediction, target, *args, **kwargs) -> torch.Tensor:
    kw = dict()
    device = kwargs["device"]
    if "weight" in kwargs:
        kw = dict(weight=kwargs["weight"].to(device))
    return F.cross_entropy(prediction.float().to(device), target.to(device), **kw)
