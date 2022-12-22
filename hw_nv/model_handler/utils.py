import torch.nn as nn


def apply_norm(model, norm_fn):
    """
        Applies norm_fn to all suitable model layers
    """
    for param in model.parameters():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Conv1d):
            print("!")
            norm_fn(param)
    return model
    

def remove_norm(model, remove_norm_fn):
    """
        Removes norm from all suitable model layers
    """
    for param in model.parameters():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Conv1d):
            remove_norm_fn(param)
    return model

def normal_init(model):
    """
        Initialise model weghts with N(0, 0.01)
    """
    for param in model.parameters():
        if isinstance(param, nn.Conv2d) or isinstance(param, nn.Conv1d):
            param.weight.data.normal_(0, 0.01)
    return model