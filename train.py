import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import TrainSet
from AdaIN import AdaINNet
from tqdm import tqdm

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--content_dir', type=str, required=True, help='content images folder path')
	parser.add_argument('--style_dir', type=str, required=True, help='style images folder path')
	parser.add_argument('--epochs', type=int, default=10, help='Number of epoch')
	parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
	parser.add_argument('--resume', type=int, default=0, help='Continue training from epoch')
	parser.add_argument('--cuda', action='store_true', help='Use CUDA')
	args = parser.parse_args()

	device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
	
	check_point_dir = './check_point/'
	weights_dir = './weights/'
	train_set = TrainSet(args.content_dir, args.style_dir)
	train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
	
	vgg_model = torch.load('vgg_normalized.pth')
	model = AdaINNet(vgg_model).to(device)

	decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-6)
	total_loss, content_loss, style_loss = 0.0, 0.0, 0.0
	losses = []
	iteration = 0

	# If resume
	if args.resume > 0:
		states = torch.load(check_point_dir + "epoch_" + str(args.resume)+'.pth')
		model.decoder.load_state_dict(states['decoder'])
		decoder_optimizer.load_state_dict(states['decoder_optimizer'])
		losses = states['losses']
		iteration = states['iteration']


	for epoch in range(args.resume + 1, args.epochs + 1):
		print("Begin epoch: %i/%i" % (epoch, int(args.epochs)))
		train_tqdm = tqdm(train_loader)
		train_tqdm.set_description('Loss: %.4f, Content loss: %.4f, Style loss: %.4f' % (total_loss, content_loss, style_loss))
		losses.append((iteration, total_loss, content_loss, style_loss))
		total_num = 0
		  
		for content_batch, style_batch in train_tqdm:
			
			decoder_optimizer.zero_grad()
			
			content_batch = content_batch.to(device)
			style_batch = style_batch.to(device)

			loss_content, loss_style = model(content_batch, style_batch)
			loss_scaled = loss_content + 10 * loss_style
			loss_scaled.backward()
			decoder_optimizer.step()
			total_loss = loss_scaled.item()
			content_loss = loss_content.item()
			style_loss = loss_style.item()

			train_tqdm.set_description('Loss: %.4f, Content loss: %.4f, Style loss: %.4f' % (total_loss, content_loss, style_loss))
			iteration += 1

			# if iteration % 100 == 0 and iteration > 0:
				
			# 	total_loss /= total_num
			# 	content_loss /= total_num
			# 	style_loss /= total_num
			# 	print('')
			# 	train_tqdm.set_description('Loss: %.4f, Content loss: %.4f, Style loss: %.4f' % (total_loss, content_loss, style_loss))
				
			# 	losses.append((iteration, total_loss, content_loss, style_loss))
				
			# 	total_loss, content_loss, style_loss = 0.0, 0.0, 0.0
			# 	total_num = 0  

			# if iteration % np.ceil(len(train_loader.dataset)/args.batch_size) == 0 and iteration > 0:
			# 	total_loss /= total_num
			# 	content_loss /= total_num
			# 	style_loss /= total_num
			# 	total_num = 0
		
		
		print('Finished epoch: %i/%i' % (epoch, int(args.epochs)))

		# states = {'decoder': model.decoder.state_dict(), 'decoder_optimizer': decoder_optimizer.state_dict(), 
		# 	'losses': losses, 'iteration': iteration}
		# torch.save(states, check_point_dir + 'epoch_%i.pth' % (epoch))
		# torch.save(model.decoder.state_dict(), weights_dir + 'decoder_epoch_%i.pth' % (epoch))	
		# np.savetxt("losses", losses, fmt='%i,%.4f,%.4f,%.4f')						

if __name__ == '__main__':
	main()







