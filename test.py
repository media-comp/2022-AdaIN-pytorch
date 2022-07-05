import os
import argparse
import torch
import time
import numpy as np
from pathlib import Path
from AdaIN import AdaINNet
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
from utils import adaptive_instance_normalization, grid_image, transform,linear_histogram_matching, Range
from glob import glob
import cv2 as cv2

parser = argparse.ArgumentParser()
parser.add_argument('--content_image', type=str, help='Content image file path')
parser.add_argument('--content_dir', type=str, help='Content image folder path')
parser.add_argument('--style_image', type=str, help='Style image file path')
parser.add_argument('--style_dir', type=str, help='Content image folder path')
parser.add_argument('--decoder_weight', type=str, default='decoder.pth', help='Decoder weight file path')
parser.add_argument('--alpha', type=float, default=1.0, choices=[Range(0.0, 1.0)], help='Alpha [0.0, 1.0] controls style transfer level')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
parser.add_argument('--grid_pth', type=str, default=None, help='Specify a grid image path (default=None) if generate a grid image that contains all style transferred images')
parser.add_argument('--color_control', action='store_true', help='Preserve content color')
parser.add_argument('--spatial_control', action='store_true', help='Preserve content color')
parser.add_argument('--mask', type=str, help='Mask image file path')

args = parser.parse_args()
assert args.content_image or args.content_dir
assert args.style_image or args.style_dir
assert args.decoder_weight

device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')


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
	
	mix_enc = alpha * transfer_enc + (1-alpha) * content_enc
	return decoder(mix_enc)


def main():	
	# Read content images and style images
	if args.content_image:
		content_pths = [Path(args.content_image)]
	else:
		content_pths = [Path(f) for f in glob(args.content_dir+'/*')]
	
	if args.style_image:
		style_pths = [Path(args.style_image)]
	else:
		style_pths = [Path(f) for f in glob(args.style_dir+'/*')]

	assert len(content_pths) > 0, 'Failed to load content image'
	assert len(style_pths) > 0, 'Failed to load style image'

	# Prepare directory for saving results
	out_dir = './results/'
	os.makedirs(out_dir, exist_ok=True)

	# Load AdaIN model
	vgg = torch.load('vgg_normalized.pth')
	model = AdaINNet(vgg).to(device)
	model.decoder.load_state_dict(torch.load(args.decoder_weight))
	model.eval()
	
	# Prepare image transform
	t = transform(512)
	
	# Prepare grid image, add style images to the first row
	if args.grid_pth:
		# Add empty image
		imgs = [np.ones((1, 1, 3), np.uint8) * 255]
		for style_pth in style_pths:
			imgs.append(Image.open(style_pth))
	
	# Timer
	times = []

	for content_pth in content_pths:
		content_img = Image.open(content_pth)
		content_tensor = t(content_img).unsqueeze(0).to(device)
		
		if args.grid_pth:
			imgs.append(content_img)

		for style_pth in style_pths:
			
			style_tensor = t(Image.open(style_pth)).unsqueeze(0).to(device)
			
			# Linear Histogram Matching if needed
			if args.color_control:
				style_tensor = linear_histogram_matching(content_tensor,style_tensor)

			# Start time
			tic = time.perf_counter()
			
			# Execute style transfer
			with torch.no_grad():
				out_tensor = style_transfer(content_tensor, style_tensor, model.encoder, model.decoder, args.alpha).cpu()
		
			 # End time
			toc = time.perf_counter()
			print("Content: " + content_pth.stem + ". Style: " \
				+ style_pth.stem + '. Alpha: ' + str(args.alpha) + '. Style Transfer time: %.4f seconds' % (toc-tic))
			times.append(toc-tic)

			# Save image
			out_pth = out_dir + content_pth.stem + '_style_' + style_pth.stem + '_alpha' + str(args.alpha)
			if args.color_control: out_pth += '_colorcontrol'
			out_pth += content_pth.suffix
			save_image(out_tensor, out_pth)

			if args.grid_pth:
				imgs.append(Image.open(out_pth))	

	# Remove runtime of first iteration because it is flawed for some unknown reason
	if len(times) > 1:
		times.pop(0)
		avg = sum(times)/len(times)
		print("Average style transfer time: %.4f seconds" % (avg))

	# Generate grid image
	if args.grid_pth:
		print("Generating grid image")
		grid_image(len(content_pths) + 1, len(style_pths) + 1, imgs, save_pth=args.grid_pth)
		print("Finished")
	#generate spatial control image, there must be only one file in the directory, and two style images in directory
	if args.spatial_control:
		onlyfiles = [f for f in os.listdir("results") if os.path.isfile(os.path.join("results", f))]
		mask_img = cv2.imread(args.mask)
		out_img=cv2.imread("results/"+onlyfiles[0])
		out_img = np.array(out_img)
		im = cv2.resize(mask_img, (512, 680))
		mask = im[:, :, 0] == 255
		reverse_mask =mask==False
		mask = np.expand_dims(mask, axis=-1)
		reverse_mask = np.expand_dims(reverse_mask, axis=-1)
		post_out_img = out_img * mask
		out_img = cv2.imread("results/" + onlyfiles[1])
		out_img = np.array(out_img)
		post_reverse_out_img=out_img * reverse_mask
		final_out_img=post_reverse_out_img+post_out_img
		cv2.imwrite("results/spatial_result.jpg",final_out_img)
		
if __name__ == '__main__':
	main()
