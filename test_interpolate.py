import os
import argparse
import torch
import time
import numpy as np
from pathlib import Path
from AdaIN import AdaINNet
from PIL import Image
from torchvision.utils import save_image
from utils import adaptive_instance_normalization, transform, Range, grid_image
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--content_image', type=str, help='Test image file path')
parser.add_argument('--style_image', type=str, required=True, help='Multiple Style image file path, separated by comma')
parser.add_argument('--decoder_weight', type=str, required=True, help='Decoder weight file path')
parser.add_argument('--alpha', type=float, default=1.0, choices=[Range(0.0, 1.0)], help='Alpha [0.0, 1.0] controls style transfer level')
parser.add_argument('--interpolation_weights', type=str, help='Weights of interpolate multiple style images')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--grid_pth', type=str, default=None, help='Specify a grid image path (default=None) if generate a grid image that contains all style transferred images. \
	if use grid mode, provide 4 style images')
args = parser.parse_args()
assert args.content_image
assert args.style_image
assert args.decoder_weight
assert args.interpolation_weights or args.grid_pth

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


def interpolate_style_transfer(content_tensor, style_tensor, encoder, decoder, alpha=1.0, interpolation_weights=None):
	"""
	Given content image and multiple style images, generate feature maps with encoder, apply 
	neural style transfer with adaptive instance normalization, interpolate style image features 
	with interpolation weights, generate output image with decoder

	Args:
		content_tensor (torch.FloatTensor): Content image 
		style_tensor (torch.FloatTensor): Multiple Style Images
		encoder: Encoder (vgg19) network
		decoder: Decoder network
		alpha (float, default=1.0): Weight of style image feature 
		interpolation_weights (list): Weight of each style image 
	
	Return:
		output_tensor (torch.FloatTensor): Interpolate Style Transfer output image
	"""
	
	content_enc = encoder(content_tensor)
	style_enc = encoder(style_tensor)
	
	transfer_enc = torch.zeros_like(content_enc).to(device)
	full_enc = adaptive_instance_normalization(content_enc, style_enc)
	for i, w in enumerate(interpolation_weights):
		transfer_enc += w * full_enc[i]

	mix_enc = alpha * transfer_enc + (1-alpha) * content_enc
	return decoder(mix_enc)
	

def main():	
	# Read content and style image
	if args.content_image:
		content_pths = [Path(args.content_image)]
	else:
		content_pths = [Path(f) for f in glob(args.content_dir+'/*')]

	style_pths_list = args.style_image.split(',')
	style_pths = [Path(pth) for pth in style_pths_list]
	
	inter_weights = []
	# If grid mode, use 4 style images, 5x5 interpolation weights
	if args.grid_pth:
		assert len(style_pths) == 4
		inter_weights = [ [ min(4-a, 4-b) / 4,  min(4-a, b) / 4, min(a, 4-b) / 4, min(a, b) / 4] \
			for a in range(5) for b in range(5) ]

	# Use user input interpolation weights
	else:
		inter_weight = [float(i) for i in args.interpolation_weights.split(',')]
		inter_weight = [i / sum(inter_weight) for i in inter_weight]
		inter_weights.append(inter_weight)
	

	out_dir = './results_interpolate/'
	os.makedirs(out_dir, exist_ok=True)
	
	# Load AdaIN model
	vgg = torch.load('vgg_normalized.pth')
	model = AdaINNet(vgg).to(device)
	model.decoder.load_state_dict(torch.load(args.decoder_weight))
	model.eval()
	
	# Prepare image transform
	t = transform(512)

	imgs = []

	for content_pth in content_pths:
		content_tensor = t(Image.open(content_pth)).unsqueeze(0).to(device)

		style_tensor = []
		for style_pth in style_pths:
			img = Image.open(style_pth)
			style_tensor.append(transform([512, 512])(img)) # Convert style images to same size
		style_tensor = torch.stack(style_tensor, dim=0).to(device)
		
		for inter_weight in inter_weights:			
			with torch.no_grad():
				out_tensor = out_tensor = interpolate_style_transfer(content_tensor, style_tensor, model.encoder, model.decoder, args.alpha, inter_weight).cpu()
			
			print("Content: " + content_pth.stem + ". Style: " + str([style_pth.stem for style_pth in style_pths]) + ". Interpolation weight: ", str(inter_weight))

			out_pth = out_dir + content_pth.stem + '_interpolate_' + str(inter_weight) + content_pth.suffix
			save_image(out_tensor, out_pth)

			if args.grid_pth:
				imgs.append(Image.open(out_pth))

	if args.grid_pth:
		print("Generating grid image")
		grid_image(5, 5, imgs, save_pth=args.grid_pth)
		print("Finished")

if __name__ == '__main__':
	main()