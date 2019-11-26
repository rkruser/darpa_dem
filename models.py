"""
Define our neural network models and optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from dataloader import L2DATA

from itertools import chain
import os

# Should I have max pools like alexnet?
class OurRewardPredictor(nn.Module):
    def __init__(self, loadmodel=False, loadfolder=L2DATA, experiment_name='gen_syllabus', 
                model_name="PongRewardPredictor", color_adversary=False):
        super().__init__()

        if loadmodel:
            print("Loading model")
            loadfile = os.path.join(loadfolder, 'models', experiment_name, model_name+'.pkl')
            f = open(loadfile, 'rb')
            self.__dict__.update(pickle.load(f))
            f.close()
        else:
            self.experiment_name = experiment_name
            self.model_name = model_name
            self.task_spaces = None
            self.current_input_spaces = None
            self.current_output_spaces = None
            self.color_adversary = color_adversary

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 5, stride=2, padding=2), # 3x128x128 --> 64x64x64 =lower((128-5+2*2)/2)+1
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 5, stride=2, padding=2), # 64x64x64 --> 128x32x32
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), # 128x32x32 --> 256x16x16
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 256x16x16 --> 256x8x8
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=2, padding=1), # 256x8x8 --> 256x4x4
                nn.ReLU(inplace=True)
                )

            self.reward_classifier = nn.Sequential(
                nn.Dropout(), #?
                nn.Linear(256*4*4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(), #?
                nn.Linear(256,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1)
                )

            self.features = self.features.to(self.device)
            self.reward_classifier = self.reward_classifier.to(self.device)

            self.main_optimizer = optim.Adam(chain(self.features.parameters(), self.reward_classifier.parameters()), 
                                            lr=0.0002) #betas=(0.9, 0.999)

            if self.color_adversary:
                self.color_discriminator = nn.Sequential(
                        nn.Dropout(), #?
                        nn.Linear(256*4*4, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(), #?
                        nn.Linear(256,256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256,1) #Just 1 for now, binary colors
                        )
                self.color_discriminator = self.color_discriminator.to(self.device)
                self.disc_optimizer = optim.Adam(self.color_discriminator.parameters(), lr=0.0002)
            else:
                self.color_discriminator = None
                self.disc_optimizer = None

    # These don't affect the rest of the model currently
    def set_task_spaces(self, task_spaces):
        self.task_spaces = task_spaces

    def set_input_space(self, input_spaces):
        self.current_input_spaces = input_spaces

    def set_output_space(self, output_spaces):
        self.current_output_spaces = output_spaces

    def forward(self, x):
        # x probably has shape (batchsize x 3 x 1 x 128 x 128)
        # x 1 is nframes
#        x = x['10_frames']
#        x = torch.Tensor(x) #From numpy array to Tensor
        x = x.to(self.device)
#        x = x[:,:,0,:,:].squeeze(2)
        x = self.features(x)
        x = x.view(x.size(0),-1)
        y = self.reward_classifier(x).squeeze()

        if self.color_adversary:
            w = x.detach() # ??
            cdetached = self.color_discriminator(w).squeeze()
            c = self.color_discriminator(x).squeeze() #don't see a way around two versions of same calculation
        else:
            cdetached = None
            c = None

        y_hat = { 'reward_pred': y, 'color_pred': c, 'adversary_pred':cdetached}
#                  'reward': y.detach().cpu().numpy()  }

        return y_hat #Question: should I make discriminator separate,
                                    # and update its weights before redoing calculation?

    def update(self, loss):
       self.zero_grad() #Does this get both models?
       mainLoss = loss['reward_loss'] #includes adversarial color loss
       mainLoss.backward()
       self.main_optimizer.step()
       if self.color_adversary:
           adversaryLoss = loss['adversary_loss']
           adversaryLoss.backward()
           self.disc_optimizer.step()

    def save_model(self, folder=L2DATA):
        fullfolder = os.path.join(folder, 'models', self.experiment_name)
        if not os.path.exists(fullfolder):
            os.makedirs(fullfolder)
        fname = os.path.join(fullfolder, self.model_name+'.pkl')
        f = open(fname, 'wb')
        pickle.dump(self.__dict__, f, 2) #pickle with protocol 2 for efficiency
        f.close()




class OurOptimizer:
    def __init__(self, model, lmbda=1):
        self.device = model.device
        self.lmbda = lmbda

#    def calculate_loss(self, y, y_hat): #y and y_hat should be dicts of all info
#        reward_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['reward_loss_pred'], 
#                        torch.Tensor(y['reward']).to(self.device))
#
#        if y_hat['color_loss_pred'] is not None and y_hat['adversary_loss_pred'] is not None:
#            color_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['color_loss_pred'], 
#                            1-torch.Tensor(y['color']))
#            adversary_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['adversary_loss_pred'], 
#                            torch.Tensor(y['color']).to(self.device))
#        else:
#            color_loss = 0
#            adversary_loss = None
#
#        reward_loss = reward_loss+self.lmbda*color_loss
#
#        loss = {'reward_loss': reward_loss, 'adversary_loss': adversary_loss}
#                 #'reward': reward_loss_grad.detach().numpy()} # numpy for benefit of TEF
#        return loss
        # loss y['reward'], y_hat['reward']
        # loss y['color'], y_hat['color'] (negative of adversary loss)
        # loss y['adversary'], y_hat['adversary']
        # add first two losses together with some constant coefficient

    def calculate_loss(self, y, y_hat):
        reward_predictions = y_hat['reward_pred']
        color_predictions = y_hat['color_pred']
        adversary_predictions = y_hat['adversary_pred']
        reward_actual = torch.Tensor(y['reward']).to(self.device)
        color_actual = torch.Tensor(y['color']).to(self.device)

        reward_loss = nn.functional.binary_cross_entropy_with_logits(reward_predictions, reward_actual)
        reward_acc = ((reward_predictions > 0).int() == reward_actual.int()).sum().float() / len(reward_actual)
        if color_predictions is not None:
            color_loss = nn.functional.binary_cross_entropy_with_logits(color_predictions, 1-color_actual)
        else:
            color_loss = 0
        if adversary_predictions is not None:
            adversary_loss = nn.functional.binary_cross_entropy_with_logits(adversary_predictions, color_actual)
            adversary_acc = ((adversary_predictions > 0).int() == color_actual.int()).sum().float() / len(color_actual)
        else:
            adversary_loss = None
            adversary_acc = None

        reward_loss = reward_loss+self.lmbda*color_loss

        losses = { 'reward_loss': reward_loss,
                   'adversary_loss': adversary_loss,
                   'reward_acc': reward_acc,
                   'adversary_acc': adversary_acc }

        return losses



        
# Next, need to design experiment json
# Figure out how to get only two colors as data
# Extract loss plots
# Do preliminary runs

class AverageMeter:
    def __init__(self):
        self.count = 0
        self.value = 0.0

    def update(self, value, num=1):
        self.count += num
        self.value += num*value #Not sure if that works for all losses, but works for accuracy

    def reset(self):
        self.count = 0
        self.value = 0.0

    def average(self):
        if self.count > 0:
            return self.value / self.count
        else:
            return 0


class Meters:
    def __init__(self, *args):
        self.meters = {}
        for name in args:
            self.add_meter(name)

    def add_meter(self, name):
        self.meters[name] = AverageMeter()

    def reset_meter(self, name):
        self.meters[name].reset()

    def average(self, name):
        return self.meters[name].average()

    def update(self, name, value, num=1):
        self.meters[name].update(value, num)

    def reset_all(self):
        for k in self.meters:
            self.reset_meter(k)
