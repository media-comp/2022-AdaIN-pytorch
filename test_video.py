import os
import argparse
import torch
from pathlib import Path
from AdaIN import AdaINNet
from PIL import Image
from utils import transform, adaptive_instance_normalization, Range
import cv2
import imageio
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--content_video', type=str, required=True, help='Content video file path')
parser.add_argument('--style_image', type=str, required=True, help='Style image file path')
parser.add_argument('--decoder_weight', type=str, default='decoder.pth', help='Decoder weight file path')
parser.add_argument('--alpha', type=float, default=1.0, choices=[Range(0.0, 1.0)], help='Alpha [0.0, 1.0] controls style transfer level')
parser.add_argument('--cuda', action='store_true', help='Use CUDA')
args = parser.parse_args()

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
	# Read video file
	content_video_pth = Path(args.content_video)
	content_video = cv2.VideoCapture(str(content_video_pth))
	style_image_pth = Path(args.style_image)
	style_image = Image.open(style_image_pth)

	# Read video info
	fps = int(content_video.get(cv2.CAP_PROP_FPS))
	frame_count = int(content_video.get(cv2.CAP_PROP_FRAME_COUNT))
	video_height = int(content_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	video_width = int(content_video.get(cv2.CAP_PROP_FRAME_WIDTH))

	# Prepare loop
	video_tqdm = tqdm(frame_count)

	# Prepare output video writer
	out_dir = './results_video/'
	os.makedirs(out_dir, exist_ok=True)
	out_pth = Path(out_dir + content_video_pth.stem + '_style_' \
		+ style_image_pth.stem + content_video_pth.suffix)
	writer = imageio.get_writer(out_pth, mode='I', fps=fps)

	# Load AdaIN model
	vgg = torch.load('vgg_normalized.pth')
	model = AdaINNet(vgg).to(device)
	model.decoder.load_state_dict(torch.load(args.decoder_weight))
	model.eval()
	
	t = transform(512)

	style_tensor = t(style_image).unsqueeze(0).to(device)


	while content_video.isOpened():
		ret, content_image = content_video.read()
		# Failed to read a frame
		if not ret:
			break
		
		content_tensor = t(Image.fromarray(content_image)).unsqueeze(0).to(device)
		
		with torch.no_grad():
			out_tensor = style_transfer(content_tensor, style_tensor, model.encoder
				, model.decoder, args.alpha).cpu().detach().numpy()
		
		# Convert output frame to original size and rgb range (0,255)
		out_tensor = np.squeeze(out_tensor, axis=0)
		out_tensor = np.transpose(out_tensor, (1, 2, 0))
		out_tensor = cv2.normalize(src=out_tensor, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		out_tensor = cv2.resize(out_tensor, (video_width, video_height), interpolation=cv2.INTER_CUBIC)

		# Write output frame to video
		writer.append_data(np.array(out_tensor))
		video_tqdm.update(1)

	content_video.release()

	print("\nContent: " + content_video_pth.stem + ". Style: " + style_image_pth.stem +'\n')

if __name__ == '__main__':
	main()