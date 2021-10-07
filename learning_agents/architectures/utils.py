def identity(x, dim=0):
    """
    Return input without any change.
    x: torch.Tensor
    :return: torch.Tensor
    """
    return x


def init_layer_uniform(layer, init_w=3e-3, init_b=0.1):
    """
    Init uniform parameters on the single layer
    layer: nn.Linear
    init_w: float = 3e-3
    :return: nn.Linear
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_b, init_b)

    return layer