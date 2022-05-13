import torch.nn as nn

vgg19_cfg = [3, 64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
decoder_cfg = [512, 256, "U", 256, 256, 256, 128, "U", 128, 64, 'U', 64, 3]

def vgg19(weights=None):
    """
    Build vgg19 network. Load weights if weights are given.

    Args:
        weights (dict): vgg19 pretrained weights

    Return:
        layers (nn.Sequential): vgg19 layers
    """

    modules = make_block(vgg19_cfg)
    modules = [nn.Conv2d(3, 3, kernel_size=1)] + list(modules.children())
    layers = nn.Sequential(*modules)

    if weights:
        layers.load_state_dict(weights)
    
    return layers


def decoder(weights=None):
    """
    Build decoder network. Load weights if weights are given.

    Args:
        weights (dict): decoder pretrained weights

    Return:
        layers (nn.Sequential): decoder layers
    """

    modules = make_block(decoder_cfg)
    layers = nn.Sequential(*list(modules.children())[:-1]) # no relu at the last layer

    if weights:
        layers.load_state_dict(weights)

    return layers


def make_block(config):
    """
    Helper function for building blocks of convolutional layers.

    Args:
        config (list): List of layer configs. "M"
            "M" - Max pooling layer. 
            "U" - Upsampling layer. 
            i (int) - Convolutional layer (i filters) plus ReLU activation. 
    Return:
        layers (nn.Sequential): block layers
    """
    layers = []
    in_channels = config[0]
    
    for c in config[1:]:
        if c == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        elif c == "U":
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        else:
            assert(isinstance(c, int))
            layers.append(nn.Conv2d(in_channels, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = c

    return nn.Sequential(*layers)
