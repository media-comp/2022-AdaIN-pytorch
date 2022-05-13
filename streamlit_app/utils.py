import os
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

def adaptive_instance_normalization(x, y, eps=1e-5):
	"""
	Adaptive Instance Normalization. Perform neural style transfer given content image x
	and style image y.

	Args:
		x (torch.FloatTensor): Content image tensor
		y (torch.FloatTensor): Style image tensor
		eps (float, default=1e-5): Small value to avoid zero division

	Return:
		output (torch.FloatTensor): AdaIN style transferred output
	"""

	mu_x = torch.mean(x, dim=[2, 3])
	mu_y = torch.mean(y, dim=[2, 3])
	mu_x = mu_x.unsqueeze(-1).unsqueeze(-1)
	mu_y = mu_y.unsqueeze(-1).unsqueeze(-1)

	sigma_x = torch.std(x, dim=[2, 3])
	sigma_y = torch.std(y, dim=[2, 3])
	sigma_x = sigma_x.unsqueeze(-1).unsqueeze(-1) + eps
	sigma_y = sigma_y.unsqueeze(-1).unsqueeze(-1) + eps

	return (x - mu_x) / sigma_x * sigma_y  + mu_y

def transform(size):
	"""
	Image preprocess transformation. Resize image and convert to tensor.

	Args:
		size (int): Resize image size

	Return:
		output (torchvision.transforms): Composition of torchvision.transforms steps
	"""
	
	t = []
	t.append(transforms.Resize(size))
	t.append(transforms.ToTensor())
	t = transforms.Compose(t)
	return t

def grid_image(row, col, images, height=6, width=6, save_pth='grid.png'):
	"""
	Generate and save an image that contains row x col grids of images.

	Args:
		row (int): number of rows
		col (int): number of columns
		images (list of PIL image): list of images.
		height (int) : height of each image (inch)
		width (int) : width of eac image (inch)
		save_pth (str): save file path
	"""

	width = col * width
	height = row * height
	plt.figure(figsize=(width, height))
	for i, image in enumerate(images):
		plt.subplot(row, col, i+1)
		plt.imshow(image)
		plt.axis('off')
		plt.subplots_adjust(wspace=0.01, hspace=0.01)
	plt.savefig(save_pth)


def linear_histogram_matching(content_tensor, style_tensor):
	"""
	Given content_tensor and style_tensor, transform style_tensor histogram to that of content_tensor.

	Args:
		content_tensor (torch.FloatTensor): Content image 
		style_tensor (torch.FloatTensor): Style Image
	
	Return:
		style_tensor (torch.FloatTensor): histogram matched Style Image
	"""
    #for batch
	for b in range(len(content_tensor)):
		std_ct = []
		std_st = []
		mean_ct = []
		mean_st = []
		#for channel
		for c in range(len(content_tensor[b])):
			std_ct.append(torch.var(content_tensor[b][c],unbiased = False))
			mean_ct.append(torch.mean(content_tensor[b][c]))
			std_st.append(torch.var(style_tensor[b][c],unbiased = False))
			mean_st.append(torch.mean(style_tensor[b][c]))
			style_tensor[b][c] = (style_tensor[b][c] - mean_st[c]) * std_ct[c] / std_st[c] + mean_ct[c]
	return style_tensor


class TrainSet(Dataset):
	"""
	Build Training dataset
	"""
	def __init__(self, content_dir, style_dir, crop_size = 256):
		super().__init__()

		self.content_files = [Path(f) for f in glob(content_dir+'/*')]
		self.style_files = [Path(f) for f in glob(style_dir+'/*')]
		
		self.transform = transforms.Compose([
			transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.RandomCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])

		Image.MAX_IMAGE_PIXELS = None
		ImageFile.LOAD_TRUNCATED_IMAGES = True
	
	def __len__(self):
		return min(len(self.style_files), len(self.content_files))

	def __getitem__(self, index):
		content_img = Image.open(self.content_files[index]).convert('RGB')
		style_img = Image.open(self.style_files[index]).convert('RGB')
	
		content_sample = self.transform(content_img)
		style_sample = self.transform(style_img)

		return content_sample, style_sample

class Range(object):
	"""
	Helper class for input argument range restriction
	"""
	def __init__(self, start, end):
		self.start = start
		self.end = end
	def __eq__(self, other):
		return self.start <= other <= self.end