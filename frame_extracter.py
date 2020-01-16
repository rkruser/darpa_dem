import os
import torch
from torchvision.utils import save_image
from dataloader import Construct_L2M_Dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--syllabus', default='experiment1.json')
args = parser.parse_args()

dataset, _ = Construct_L2M_Dataset(args.syllabus, train_proportion=1, resize=None)

size = len(dataset)
print(size)

'''
first_im, _ = dataset[0]
mid_im, _ = dataset[size//2]
last_im, _ = dataset[-1]

save_image(first_im, './latex/figures/'+args.syllabus+'_first_im.png')
save_image(mid_im, './latex/figures/'+args.syllabus+'_mid_im.png')
save_image(last_im, './latex/figures/'+args.syllabus+'_last_im.png')
'''

for i in range(10):
    im, _ = dataset[i]
    save_image(im, './latex/figures/'+args.syllabus+'_im'+str(i)+'.png')












