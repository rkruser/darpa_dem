import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import learnkit
from learnkit.utils import module_relative_file

from models import OurRewardPredictor, OurOptimizer
from dataloader import L2M_Pytorch_Dataset, L2DATA




def train_model(gen_syllabus, nepochs):
    exp_name = os.path.basename(os.path.splitext(gen_syllabus)[0])
    model = OurRewardPredictor(loadmodel=False, experiment_name=exp_name, color_adversary=False)
    optim = OurOptimizer(model)
    loader = DataLoader(L2M_Pytorch_Dataset(gen_syllabus), batch_size=32, shuffle=True, num_workers=4)

    for epoch in range(nepochs):
        print(epoch)
        for i, pt in enumerate(loader):
            #print("   ", i)
            x = pt[0]
            y = {'reward': pt[1], 'color': pt[2]}

            y_hat = model.forward(x)
            loss = optim.calculate_loss(y,y_hat)
            model.update(loss)

    model.save_model() 








if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--gen_syllabus', default='gen_syllabus.json')
   parser.add_argument('--train_syllabus', default='train_predict_total_reward_syllabus.json')
   parser.add_argument('--random_seed', type=int, default=1234)
   parser.add_argument('--nepochs', type=int, default=10)
   args = parser.parse_args()
   torch.manual_seed(args.random_seed)
   train_model(module_relative_file(__file__, args.gen_syllabus), args.nepochs)
