import torch
import torch.nn as nn
from Network import vgg19, decoder
from utils import adaptive_instance_normalization

class AdaINNet(nn.Module):
    """
    AdaIN Style Transfer Network

    Args:
        vgg_weight: pretrained vgg19 weight
    """
    def __init__(self, vgg_weight):
        super().__init__()
        self.encoder = vgg19(vgg_weight)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:22]) # drop layers after 4_1
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        
        self.decoder = decoder()
        
        self.mseloss = nn.MSELoss()

    def _style_loss(self, x, y):
        return self.mseloss(torch.mean(x, dim=[2, 3]), torch.mean(y, dim=[2, 3])) + \
            self.mseloss(torch.std(x, dim=[2, 3]), torch.std(y, dim=[2, 3]))

    def forward(self, content, style, alpha=1.0):
        content_enc = self.encoder(content)
        style_enc = self.encoder(style)
        transfer_enc = adaptive_instance_normalization(content_enc, style_enc)

        out = self.decoder(transfer_enc)
        
        # vgg19 layer relu1_1
        style_relu11 = self.encoder[:3](style)
        out_relu11 = self.encoder[:3](out)

        # vgg19 layer relu2_1
        style_relu21 = self.encoder[3:8](style_relu11)
        out_relu21 = self.encoder[3:8](out_relu11)

        # vgg19 layer relu3_1
        style_relu31 = self.encoder[8:13](style_relu21)
        out_relu31 = self.encoder[8:13](out_relu21)

        # vgg19 layer relu4_1
        out_enc = self.encoder[13:](out_relu31)

        content_loss = self.mseloss(out_enc, transfer_enc)
        style_loss = self._style_loss(out_relu11, style_relu11) + self._style_loss(out_relu21, style_relu21) + \
            self._style_loss(out_relu31, style_relu31) + self._style_loss(out_enc, style_enc)

        return content_loss, style_loss