"""
Define our neural network models and optimizers
"""

import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain

# Should I have max pools like alexnet?
class OurRewardPredictor(nn.Module):
    def __init__(self, color_adversary=False):
        super(OurRewardPredictor, self).__init__()

        self.task_spaces = None
        self.current_input_spaces = None
        self.current_output_spaces = None

        self.color_adversary = color_adversary
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
        x = x['10_frames']
        x = torch.Tensor(x) #From numpy array to Tensor
        x = x[:,:,0,:,:].squeeze(2)
        x = self.features(x)
        x = x.view(x.size(0),-1)
        y = self.reward_classifier(x)

        if self.color_adversary:
            w = x.detach() # ??
            cdetached = self.color_discriminator(w)
            c = self.color_discriminator(x) #don't see a way around two versions of same calculation
        else:
            cdetached = None
            c = None

        y_hat = { 'reward': y, 'color': c, 'adversary':cdetached }

        return y_hat #Question: should I make discriminator separate,
                                    # and update its weights before redoing calculation?

    def update(self, loss):
       mainLoss = loss['reward_loss_grad'] #includes adversarial color loss
       mainLoss.backward()
       self.main_optimizer.step()
       if self.color_adversary:
           adversaryLoss = loss['adversary_loss_grad']
           adversaryLoss.backward()
           self.disc_optimizer.step()


class OurOptimizer:
    def __init__(self, lmbda=1):
        self.lmbda = lmbda

    def calculate_loss(self, y, y_hat): #y and y_hat should be dicts of all info
        reward_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['reward'], torch.Tensor(y['reward']))

        if y['color'] is not None and y['adversary'] is not None:
            color_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['color'], 1-torch.Tensor(y['color']))
            adversary_loss = nn.functional.binary_cross_entropy_with_logits(y_hat['adversary'], torch.Tensor(y['adversary']))
        else:
            color_loss = 0
            adversary_loss = None

        reward_loss_grad = reward_loss+self.lmbda*color_loss

        loss = {'reward_loss_grad': reward_loss_grad, 'adversary_loss_grad': adversary_loss,
                 'reward': reward_loss_grad.detach().numpy()} # numpy for benefit of TEF
        return loss
        # loss y['reward'], y_hat['reward']
        # loss y['color'], y_hat['color'] (negative of adversary loss)
        # loss y['adversary'], y_hat['adversary']
        # add first two losses together with some constant coefficient


        
# Next, need to design experiment json
# Figure out how to get only two colors as data
# Extract loss plots
# Do preliminary runs


