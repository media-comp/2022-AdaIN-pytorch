import torch
from PIL import Image
from torchvision.utils import save_image
from utils import transform
content_path = './images/content/brad_pitt.jpg'
style_path = './images/art/flower_of_life.jpg'
output_path = './results/output_1.jpg'

#Prepare image transform
t = transform(512)

content_image = Image.open(content_path)
style_image = Image.open(style_path)
content_tensor = t(content_image).unsqueeze(0)
style_tensor = t(style_image).unsqueeze(0)
#content_tensor = torch.tensor([[[[2.0,2.0,2.0],[4.0,4.0,4.0],[3.0,3.0,3.0]],[[5.0,5.0,5.0],[3.0,3.0,3.0],[1.0,1.0,1.0]],[[3.0,3.0,3.0],[1.0,1.0,1.0],[2.0,2.0,2.0]]]])
print(content_tensor)
print(content_tensor.shape)
std_ct_1, mean_ct_1 = torch.var_mean(content_tensor[0][0],unbiased = False)
std_ct_2, mean_ct_2 = torch.var_mean(content_tensor[0][1],unbiased = False)
std_ct_3, mean_ct_3 = torch.var_mean(content_tensor[0][2],unbiased = False)
std_st_1, mean_st_1 = torch.var_mean(style_tensor[0][0],unbiased = False)
std_st_2, mean_st_2 = torch.var_mean(style_tensor[0][1],unbiased = False)
std_st_3, mean_st_3 = torch.var_mean(style_tensor[0][2],unbiased = False)
style_tensor[0][0] = (style_tensor[0][0] - mean_st_1) * std_ct_1 / std_st_1 + mean_ct_1
style_tensor[0][1] = (style_tensor[0][1] - mean_st_2) * std_ct_2 / std_st_2 + mean_ct_2
style_tensor[0][2] = (style_tensor[0][2] - mean_st_3) * std_ct_3 / std_st_3 + mean_ct_3
#print(content_tensor)
output_tensor = style_tensor
save_image(output_tensor,output_path)