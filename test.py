import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import learnkit
from learnkit.utils import module_relative_file

from models import OurRewardPredictor, OurSimpleRewardPredictor, OurOptimizer, Meters
from dataloader import L2DATA, Construct_L2M_Dataset

#from torch.utils.tensorboard import SummaryWriter




def test_model(mclass, test_syllabus, model_name, use_adversary, resize, noise):
    exp_name = os.path.basename(os.path.splitext(test_syllabus)[0]).split('_')[0]
    model = mclass(loadmodel=True, experiment_name=exp_name, model_name=model_name,
                                color_adversary=use_adversary)
    model.train(False)
    optim = OurOptimizer(model)

#    totalset = L2M_Pytorch_Dataset(syllabus)
#    train_length = int(0.8*len(totalset))
#    test_length = len(totalset)-train_length
#    trainset, testset = torch.utils.data.random_split(totalset, (train_length, test_length))
    testset, _ = Construct_L2M_Dataset(test_syllabus, train_proportion=1, resize=resize, noise=noise)

    print("Test stats:")
    testset.print_statistics()
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    meters = Meters('testacc','testloss', 'testgrid', 'testacc_adv', 'testloss_adv')
    meters.initialize_meter('testgrid', torch.zeros(2,2), torch.zeros(2,2))

    # Test
    print("  Test")
    for i, pt in enumerate(testloader):
        x, y = pt
        batch_size = len(x)
    
        y_hat = model.forward(x)
        loss = optim.calculate_loss(y,y_hat)
    
        # Logging
        meters.update('testloss', loss['reward_loss'].item(), batch_size)
        meters.update('testacc', loss['reward_num_correct'].item(), batch_size)
        meters.update('testgrid', loss['sumgrid'], loss['totalgrid'])
        if loss['adversary_loss'] is not None:
            meters.update('testloss_adv', loss['adversary_loss'].item(), batch_size)
            meters.update('testacc_adv', loss['adversary_num_correct'].item(), batch_size)


    print("Loss:", meters.average('testloss'))
    print("Accuracy:", meters.average('testacc'))
    
    test_grid_average = meters.average('testgrid')
    print("Test grid")
    print(test_grid_average)

    if use_adversary:
        print("Test loss adversary", meters.average('testloss_adv'))
        print("Test accuracy adversary", meters.average('testacc_adv'))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_syllabus', default='experiment1_test.json')
#    parser.add_argument('--train_syllabus', default='train_predict_total_reward_syllabus.json')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--model_name', default='PongRewardsEpoch100')
    parser.add_argument('--use_adversary', action='store_true')
    parser.add_argument('--model_class', default='OurRewardPredictor')
    parser.add_argument('--noise', type=float, default=None)
     
    args = parser.parse_args()
    
    mclass = OurRewardPredictor
    resize=None
    if args.model_class == 'OurSimpleRewardPredictor':
        mclass = OurSimpleRewardPredictor
        resize = 32  

    torch.manual_seed(args.random_seed)
    test_model(mclass, module_relative_file(__file__, args.test_syllabus),
                args.model_name, 
                args.use_adversary,
                resize,
                args.noise)
