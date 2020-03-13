import os
import torch
from torchvision.utils import save_image
from dataloader import Construct_L2M_Dataset
import imageio

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--syllabus', default='experiment1.json')
args = parser.parse_args()

dataset, _ = Construct_L2M_Dataset(args.syllabus, train_proportion=1, resize=None)

size = len(dataset)
print(size)

first_im, _, _, _ = dataset[0]
mid_im, _, _, _ = dataset[size//2]
last_im, _, _, _ = dataset[-1]

syllabus = os.path.splitext(os.path.basename(args.syllabus))[0]

#first_im = 255*first_im
#print(first_im[:,50,50],first_im[:,55,50],first_im[:,65,65],first_im[:,65,70])

'''
M = torch.Tensor([[1.9615,0,0],[0.5708,2.5213,-0.9898],[-1.5472,-0.1236,2.5460]])
print(M)

first_im = first_im.permute([1,2,0]) # Color in last dim
first_im = torch.matmul(first_im,M)
first_im = first_im.permute([2,0,1])
save_image(first_im, './latex/figures/'+syllabus+'_first_im.png')
'''

'''
save_image(first_im, './latex/figures/'+syllabus+'_first_im.png')
save_image(mid_im, './latex/figures/'+syllabus+'_mid_im.png')
save_image(last_im, './latex/figures/'+syllabus+'_last_im.png')
'''

'''
for i in range(10):
    im, _ = dataset[i]
    save_image(im, './latex/figures/images/'+args.syllabus+'_im'+str(i)+'.png')
'''












