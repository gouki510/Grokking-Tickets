
def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "weight_mask" in name:
            yield buf

def parameters(model):
    r"""Returns an iterator over models trainable parameters, yielding just the
    parameter tensor.
    """
    for module in model.modules():
        for param in module.parameters(recurse=False):
            yield param

def masked_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in model.modules():
        for mask, param in zip(masks(module), module.parameters(recurse=False)):
                yield mask, param