import torch.nn as nn


def apply_norm(model, norm_fn):
    """
        Applies norm_fn to all suitable model layers
    """
    for param in model.parameters():
        if isinstance(param, nn.Conv2d):
            norm_fn(param)
    return model
    

def remove_norm(model, remove_norm_fn):
    """
        Removes norm from all suitable model layers
    """
    for param in model.named_parameters():
        if isinstance(param, nn.Conv2d):
            remove_norm_fn(param)
    return model