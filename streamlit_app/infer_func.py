import torch
import torchvision.transforms
from PIL import Image

from AdaIN import AdaINNet
from utils import adaptive_instance_normalization, transform, linear_histogram_matching

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def style_transfer(content_tensor, style_tensor, encoder, decoder, alpha=1.0):
    """
    Given content image and style image, generate feature maps with encoder, apply
    neural style transfer with adaptive instance normalization, generate output image
    with decoder

    Args:
        content_tensor (torch.FloatTensor): Content image
        style_tensor (torch.FloatTensor): Style Image
        encoder: Encoder (vgg19) network
        decoder: Decoder network
        alpha (float, default=1.0): Weight of style image feature

    Return:
        output_tensor (torch.FloatTensor): Style Transfer output image
    """

    content_enc = encoder(content_tensor)
    style_enc = encoder(style_tensor)

    transfer_enc = adaptive_instance_normalization(content_enc, style_enc)

    mix_enc = alpha * transfer_enc + (1 - alpha) * content_enc
    return decoder(mix_enc)


def convert(content_path, style_path, vgg_weights_path, decoder_weights_path, alpha, color_control):

    vgg = torch.load(vgg_weights_path)
    model = AdaINNet(vgg).to(device)
    model.decoder.load_state_dict(torch.load(decoder_weights_path))
    model.eval()

    # Prepare image transform
    t = transform(512)

    # load images
    content_img = Image.open(content_path)
    content_tensor = t(content_img).unsqueeze(0).to(device)
    style_tensor = t(Image.open(style_path)).unsqueeze(0).to(device)

    if color_control:
        style_tensor = linear_histogram_matching(content_tensor, style_tensor)

    with torch.no_grad():
        out_tensor = style_transfer(content_tensor, style_tensor, model.encoder, model.decoder, alpha).cpu()

    outimage_fname = 'output.png'
    torchvision.utils.save_image(out_tensor.squeeze(0), outimage_fname)

    return outimage_fname
