import torch.nn as nn


def apply_norm(model, norm_fn):
    """
        Applies norm_fn to all suitable model layers
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            norm_fn(layer)
    return model
    

def remove_norm(model, remove_norm_fn):
    """
        Removes norm from all suitable model layers
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            remove_norm_fn(layer)
    return model

def normal_init(model):
    """
        Initialise model weghts with N(0, 0.01)
    """
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            layer.weight.data.normal_(0, 0.01)
    return model