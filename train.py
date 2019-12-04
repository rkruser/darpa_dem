import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import learnkit
from learnkit.utils import module_relative_file

from models import OurRewardPredictor, OurOptimizer, Meters
from dataloader import L2M_Pytorch_Dataset, L2DATA

from torch.utils.tensorboard import SummaryWriter




def train_model(syllabus, nepochs, model_name, use_adversary):
    writer = SummaryWriter()
    exp_name = os.path.basename(os.path.splitext(syllabus)[0])
    model = OurRewardPredictor(loadmodel=False, experiment_name=exp_name, model_name=model_name,
                                color_adversary=use_adversary)
    optim = OurOptimizer(model)

    totalset = L2M_Pytorch_Dataset(syllabus)
    train_length = int(0.8*len(totalset))
    test_length = len(totalset)-train_length
    trainset, testset = torch.utils.data.random_split(totalset, (train_length, test_length))
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    meters = Meters('trainacc', 'testacc', 'trainloss', 'testloss', 'traingrid', 'testgrid',
        'trainacc_adv', 'trainloss_adv', 'testacc_adv', 'testloss_adv')#, 'traingrid_adv', 'testgrid_adv')
    meters.initialize_meter('traingrid', torch.zeros(2,2), torch.zeros(2,2))
    meters.initialize_meter('testgrid', torch.zeros(2,2), torch.zeros(2,2))
#    meters.initialize_meter('traingrid_adv') = np.zeros((2,2))
#    meters.initialize_meter('testgrid_adv') = np.zeros((2,2))

    writer = SummaryWriter()
    for epoch in range(nepochs):
        print(epoch)
        meters.reset_all()
        # Train
        print("  Train")
        for i, pt in enumerate(trainloader):
            #print("   ", i)
#            x = pt[0]
#            y = {'reward': pt[1], 'color': pt[2]}
            x, y = pt
            batch_size = len(x)

            y_hat = model.forward(x)
            loss = optim.calculate_loss(y,y_hat)
            model.update(loss)

            # Logging
            meters.update('trainloss', loss['reward_loss'].item(), batch_size)
            meters.update('trainacc', loss['reward_acc'].item(), batch_size)
            meters.update('traingrid', loss['proportiongrid'], loss['totalgrid'])
            if loss['adversary_loss'] is not None:
                meters.update('trainloss_adv', loss['adversary_loss'].item(), batch_size)
                meters.update('trainacc_adv', loss['adversary_acc'].item(), batch_size)

        # Test
        print("  Test")
        for i, pt in enumerate(testloader):
#            x = pt[0]
#            y = {'reward': pt[1], 'color': pt[2]}
            x, y = pt
            batch_size = len(x)

            y_hat = model.forward(x)
            loss = optim.calculate_loss(y,y_hat)

            # Logging
            meters.update('testloss', loss['reward_loss'].item(), batch_size)
            meters.update('testacc', loss['reward_acc'].item(), batch_size)
            meters.update('testgrid', loss['proportiongrid'], loss['totalgrid'])
            if loss['adversary_loss'] is not None:
                meters.update('testloss_adv', loss['adversary_loss'].item(), batch_size)
                meters.update('testacc_adv', loss['adversary_acc'].item(), batch_size)

        # Tensorboard logging
        writer.add_scalar('loss/main/train', meters.average('trainloss'), epoch)
        writer.add_scalar('accuracy/main/train', meters.average('trainacc'), epoch)
        writer.add_scalar('loss/main/test', meters.average('testloss'), epoch)
        writer.add_scalar('accuracy/main/test', meters.average('testacc'), epoch)

        train_grid_average = meters.average('traingrid')
        test_grid_average = meters.average('testgrid')
        for i in range(len(train_grid_average)):
            for j in range(len(train_grid_average[i])):
                writer.add_scalar('train/grid_{0}_{1}'.format(i,j), train_grid_average[i,j].item())
                writer.add_scalar('test/grid_{0}_{1}'.format(i,j), test_grid_average[i,j].item())

        writer.add_scalar('loss/adv/train', meters.average('trainloss_adv'), epoch)
        writer.add_scalar('accuracy/adv/train', meters.average('trainacc_adv'), epoch)
        writer.add_scalar('loss/adv/test', meters.average('testloss_adv'), epoch)
        writer.add_scalar('accuracy/adv/test', meters.average('testacc_adv'), epoch)

            

    model.save_model() 


if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('--syllabus', default='gen_syllabus.json')
   parser.add_argument('--train_syllabus', default='train_predict_total_reward_syllabus.json')
   parser.add_argument('--random_seed', type=int, default=1234)
   parser.add_argument('--nepochs', type=int, default=10)
   parser.add_argument('--model_name', default='PongRewardPredictor')
   parser.add_argument('--use_adversary', action='store_true')
   args = parser.parse_args()
   torch.manual_seed(args.random_seed)
   train_model(module_relative_file(__file__, args.syllabus), args.nepochs, 
                args.model_name, args.use_adversary)
