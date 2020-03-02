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

from dataloader import stat_grid

# Should I have max pools like alexnet?
class OurRewardPredictor(nn.Module):
    def __init__(self, loadmodel=False, loadfolder=L2DATA, experiment_name='experiment1', 
                model_name="PongRewardsEpoch100", color_adversary=False, paddle_predictor=False,
                gpuid=0, loadname=None):
        super().__init__()

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

#            self.main_optimizer = optim.Adam(chain(self.features.parameters(), self.reward_classifier.parameters()), 
#                                            lr=0.0002) #betas=(0.9, 0.999)

        if color_adversary:
            self.color_discriminator = nn.Sequential(
                    nn.Dropout(), #?
                    nn.Linear(256*4*4, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(), #?
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,1) #Just 1 for now, binary colors
                    )
    #                self.color_discriminator = self.color_discriminator.to(self.device)
    #                self.disc_optimizer = optim.Adam(self.color_discriminator.parameters(), lr=0.0002)
        else:
            self.color_discriminator = None
            self.disc_optimizer = None

        if paddle_predictor:
            self.paddle_predictor = nn.Sequential(
                    nn.Dropout(), #?
                    nn.Linear(256*4*4, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(), #?
                    nn.Linear(256,256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256,1) #Just 1 for now, binary colors
                    )
        else:
                self.paddle_predictor=None
                self.paddle_optimizer=None


        if loadmodel:
            if loadname is None:
                loadname = model_name
                
            loadfile = os.path.join(loadfolder, 'models', experiment_name, loadname+'.pkl')
            print("Loading model from {}".format(loadfile))
            with open(loadfile, 'rb') as f:
                d = pickle.load(f)
                self.features = d['_modules']['features']
                self.reward_classifier = d['_modules']['reward_classifier']
                if 'color_discriminator' in d['_modules']:
                    self.color_discriminator = d['_modules']['color_discriminator']
                if 'paddle_predictor' in d['_modules']:
                    self.paddle_predictor = d['_modules']['paddle_predictor']
                #self.__dict__.update(d)

#        self.experiment_name = experiment_name
        self.experiment_name = experiment_name.split('_')[0]
        self.model_name = model_name
        self.task_spaces = None
        self.current_input_spaces = None
        self.current_output_spaces = None
        self.color_adversary = color_adversary
#        self.paddle_predictor = paddle_predictor

        # This code creates new optimizers instead of using saved ones. Good or bad?
        self.device = torch.device("cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu")
        self.features = self.features.to(self.device)
        self.reward_classifier = self.reward_classifier.to(self.device)
        self.main_optimizer = optim.Adam(chain(self.features.parameters(), 
                                         self.reward_classifier.parameters()), 
                                         lr=0.0002)
        if self.color_adversary:
            self.color_discriminator = self.color_discriminator.to(self.device)    
            self.disc_optimizer = optim.Adam(self.color_discriminator.parameters(), lr=0.0002)

        if self.paddle_predictor:
            self.paddle_predictor = self.paddle_predictor.to(self.device)
            self.paddle_optimizer = optim.Adam(self.paddle_predictor.parameters(), lr=0.0002)



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

        if self.paddle_predictor:
            z = x.detach()
            pdetached = self.paddle_predictor(z).squeeze()
            p = self.paddle_predictor(x).squeeze()
        else:
            pdetached = None
            p = None


        y_hat = { 'reward_pred': y, 
                  'color_pred': c, 
                  'adversary_pred':cdetached,
                  'paddle_pred':p,
                  'paddle_detached_pred':pdetached } 


        return y_hat #Question: should I make discriminator separate,
                                    # and update its weights before redoing calculation?

    def update(self, loss):
       self.features.zero_grad()
       self.reward_classifier.zero_grad()
       
       ####
       self.paddle_predictor.zero_grad()

       mainLoss = loss['reward_loss'] #includes adversarial color loss
       mainLoss.backward()
       self.main_optimizer.step()

       ####
       self.paddle_optimizer.step()
       
       if self.color_adversary:
           self.color_discriminator.zero_grad()
           adversaryLoss = loss['adversary_loss']
           adversaryLoss.backward()
           self.disc_optimizer.step()
#       if self.paddle_predictor:
#           self.paddle_predictor.zero_grad()
#           paddle_loss = loss['paddle_detached_loss']
#           paddle_loss.backward()
#           self.paddle_optimizer.step()

    def save_model(self, folder=L2DATA, ask_rename=False):
        fullfolder = os.path.join(folder, 'models', self.experiment_name)
        if not os.path.exists(fullfolder):
            os.makedirs(fullfolder)
        fname = os.path.join(fullfolder, self.model_name+'.pkl')
        iters = 1
        while os.path.exists(fname):
            print("Warning: model file already exists, press enter to overwrite or type a new model name")
            if ask_rename:
                newname = input()
            else:
                newname = fname+'_'+str(iters)
                iters += 1
            if len(newname) > 0:
                self.model_name = newname
                fname = os.path.join(fullfolder, self.model_name+'.pkl')
            else:
                break
            
        f = open(fname, 'wb')
        pickle.dump(self.__dict__, f, 2) #pickle with protocol 2 for efficiency
        f.close()



        
        
# assume 3x32x32
class OurSimpleRewardPredictor(nn.Module):
    def __init__(self, loadmodel=False, loadfolder=L2DATA, experiment_name='experiment1', 
                model_name="PongRewardsEpoch100", color_adversary=False, paddle_predictor=False, gpuid=0, loadname=None):
        super().__init__()
        print("Simple reward predictor")

        if loadmodel:
            if loadname is None:
                loadname = model_name
            loadfile = os.path.join(loadfolder, 'models', experiment_name, loadname+'.pkl')
            print("Loading model from {}".format(loadfile))
            with open(loadfile, 'rb') as f:
                self.__dict__.update(pickle.load(f))
        else:

            self.features = nn.Sequential(
                nn.Linear(3*32*32, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True)
                )

            self.reward_classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256,1)
                )

#            self.features = self.features.to(self.device)
#            self.reward_classifier = self.reward_classifier.to(self.device)

#            self.main_optimizer = optim.Adam(chain(self.features.parameters(), self.reward_classifier.parameters()), 
#                                            lr=0.0002) #betas=(0.9, 0.999)

            if color_adversary:
                self.color_discriminator = nn.Sequential(
                        nn.Dropout(), #?
                        nn.Linear(256, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256,1) #Just 1 for now, binary colors
                        )
#                self.color_discriminator = self.color_discriminator.to(self.device)
#                self.disc_optimizer = optim.Adam(self.color_discriminator.parameters(), lr=0.0002)
            else:
                self.color_discriminator = None
                self.disc_optimizer = None

            if paddle_predictor:
                self.paddle_predictor = nn.Sequential(
                        nn.Dropout(), #?
                        nn.Linear(256, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256,1) #Just 1 for now, binary colors
                        )


        self.experiment_name = experiment_name
        self.model_name = model_name
        self.task_spaces = None
        self.current_input_spaces = None
        self.current_output_spaces = None
        self.color_adversary = color_adversary
        self.paddle_predictor = paddle_predictor

        self.device = torch.device("cuda:{}".format(gpuid) if torch.cuda.is_available() else "cpu")
        self.features = self.features.to(self.device)
        self.reward_classifier = self.reward_classifier.to(self.device)
        self.main_optimizer = optim.Adam(chain(self.features.parameters(), self.reward_classifier.parameters()), 
                                          lr=0.0002) #betas=(0.9, 0.999)
        if self.color_adversary:
            self.color_discriminator = self.color_discriminator.to(self.device)
            self.disc_optimizer = optim.Adam(self.color_discriminator.parameters(), lr=0.0002)

        if self.paddle_predictor:
            self.paddle_predictor = self.paddle_predictor.to(self.device)
            self.paddle_optimizer = optim.Adam(self.paddle_predictor.parameters(), lr=0.0002)
           


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
        x = x.view(x.size(0),-1)
#        x = x[:,:,0,:,:].squeeze(2)
        x = self.features(x)
        #x = x.view(x.size(0),-1)
        y = self.reward_classifier(x).squeeze()

        if self.color_adversary:
            w = x.detach() # ??
            cdetached = self.color_discriminator(w).squeeze()
            c = self.color_discriminator(x).squeeze() #don't see a way around two versions of same calculation
        else:
            cdetached = None
            c = None

        if self.paddle_predictor:
            z = x.detach()
            pdetached = self.paddle_predictor(z).squeeze()
            p = self.paddle_predictor(x).squeeze()
        else:
            pdetached=None
            p=None

        y_hat = { 'reward_pred': y, 'color_pred': c, 'adversary_pred':cdetached, 
                    'paddle_pred':p,
                    'paddle_detached_pred':pdetached } # Not sure if detached is necessary here
#                  'reward': y.detach().cpu().numpy()  }

        return y_hat #Question: should I make discriminator separate,
                                    # and update its weights before redoing calculation?

    def update(self, loss):
       self.features.zero_grad()
       self.reward_classifier.zero_grad()
       mainLoss = loss['reward_loss'] #includes adversarial color loss
       mainLoss.backward()
       self.main_optimizer.step()
       if self.color_adversary:
           self.color_discriminator.zero_grad()
           adversaryLoss = loss['adversary_loss']
           adversaryLoss.backward()
           self.disc_optimizer.step()

    def save_model(self, folder=L2DATA):
        fullfolder = os.path.join(folder, 'models', self.experiment_name)
        if not os.path.exists(fullfolder):
            os.makedirs(fullfolder)
        fname = os.path.join(fullfolder, self.model_name+'.pkl')
        while os.path.exists(fname):
            print("Warning: model file already exists, press enter to overwrite or type a new model name")
            newname = input()
            if len(newname) > 0:
                self.model_name = newname
                fname = os.path.join(fullfolder, self.model_name+'.pkl')
            else:
                break
            
        f = open(fname, 'wb')
        pickle.dump(self.__dict__, f, 2) #pickle with protocol 2 for efficiency
        f.close()       

# "Extrapolation networks" idea - exploit linearity of sub-attributes like color

class OurOptimizer:
    def __init__(self, model, lmbda=1):
        self.device = model.device
        self.lmbda = lmbda

    def calculate_loss(self, y, y_hat):
        reward_predictions = y_hat['reward_pred']
        color_predictions = y_hat['color_pred']
        adversary_predictions = y_hat['adversary_pred']
        reward_actual = torch.Tensor(y['reward']).to(self.device)
        color_actual = torch.Tensor(y['bg_color']).to(self.device)
        size_actual = torch.Tensor(y['bot/paddle/width']).to(self.device)

        reward_loss = nn.functional.binary_cross_entropy_with_logits(reward_predictions, reward_actual)
        reward_num_correct = ((reward_predictions > 0).int() == reward_actual.int()).sum().float()
        reward_acc = reward_num_correct / len(reward_actual)

        sumgrid, totalgrid, proportiongrid = stat_grid(torch.sigmoid(reward_predictions), size_actual, color_actual)

        if color_predictions is not None:
            color_loss = nn.functional.binary_cross_entropy_with_logits(color_predictions, 1-color_actual)
        else:
            color_loss = 0
        if adversary_predictions is not None:
            adversary_loss = nn.functional.binary_cross_entropy_with_logits(adversary_predictions, color_actual)
            adversary_num_correct = ((adversary_predictions > 0).int() == color_actual.int()).sum().float()
            adversary_acc = adversary_num_correct / len(color_actual)
        else:
            adversary_loss = None
            adversary_acc = None
            adversary_num_correct = None

        reward_loss = reward_loss+self.lmbda*color_loss

        losses = { 'reward_loss': reward_loss,
                   'adversary_loss': adversary_loss,
                   'reward_acc': reward_acc,
                   'reward_num_correct': reward_num_correct,
                   'adversary_acc': adversary_acc,
                   'adversary_num_correct': adversary_num_correct,
                   'sumgrid': sumgrid.to('cpu'),
                   'totalgrid': totalgrid.to('cpu'),
                   'proportiongrid': proportiongrid.to('cpu'),
                   }

        return losses


class OptWithPaddleLoss:
    def __init__(self, model, lmbda=[1,1]):
        self.device = model.device
        self.lmbda = lmbda

    def calculate_loss(self, y, y_hat):
        reward_predictions = y_hat['reward_pred']
        color_predictions = y_hat['color_pred']
        paddle_predictions = y_hat['paddle_pred']
        paddle_detached_predictions = y_hat['paddle_detached_pred']
        adversary_predictions = y_hat['adversary_pred']

        reward_actual = torch.Tensor(y['reward']).to(self.device)
        color_actual = torch.Tensor(y['bg_color']).to(self.device)
        paddle_bot_actual = torch.Tensor(y['bot/paddle/width']).to(self.device)
        paddle_agent_actual = torch.Tensor(y['agent/paddle/width']).to(self.device)

        reward_loss = nn.functional.binary_cross_entropy_with_logits(reward_predictions, reward_actual)
        reward_num_correct = ((reward_predictions > 0).int() == reward_actual.int()).sum().float()
        reward_acc = reward_num_correct / len(reward_actual)

        if paddle_predictions is not None:
             paddle_loss = nn.functional.binary_cross_entropy_with_logits(paddle_predictions, paddle_agent_actual)
        else:
            paddle_loss = 0

        if paddle_detached_predictions is not None:
            paddle_detached_loss = nn.functional.binary_cross_entropy_with_logits(paddle_detached_predictions, paddle_agent_actual)
            paddle_num_correct = ((paddle_detached_predictions > 0).int() == paddle_agent_actual.int()).sum().float()
            paddle_acc = paddle_num_correct / len(paddle_agent_actual)

            paddle_detached_rmse = None #unused for now
        else:
            paddle_detached_loss = None
            paddle_acc = None
            paddle_num_correct = None

            paddle_detached_rmse = None


        if color_predictions is not None:
            color_loss = nn.functional.binary_cross_entropy_with_logits(color_predictions, 1-color_actual)
        else:
            color_loss = 0

        if adversary_predictions is not None:
            adversary_loss = nn.functional.binary_cross_entropy_with_logits(adversary_predictions, color_actual)
            adversary_num_correct = ((adversary_predictions > 0).int() == color_actual.int()).sum().float()
            adversary_acc = adversary_num_correct / len(color_actual)
        else:
            adversary_loss = None
            adversary_acc = None
            adversary_num_correct = None

        reward_loss = reward_loss+self.lmbda[0]*color_loss+self.lmbda[1]*paddle_loss

        losses = { 'reward_loss': reward_loss,
                   'adversary_loss': adversary_loss,
                   'paddle_detached_loss': paddle_detached_loss,
                   'paddle_detached_rmse': paddle_detached_rmse,
                   'reward_acc': reward_acc,
                   'reward_num_correct': reward_num_correct,
                   'adversary_acc': adversary_acc,
                   'adversary_num_correct': adversary_num_correct,
                   'paddle_acc': paddle_acc,
                   'paddle_num_correct':paddle_num_correct,
                   'sumgrid': None,
                   'totalgrid': None,
                   'proportiongrid': None,
                   }

        return losses


        
# Next, need to design experiment json
# Figure out how to get only two colors as data
# Extract loss plots
# Do preliminary runs

class AverageMeter:
    def __init__(self):
        self.count_init_val = 0
        self.count = self.count_init_val
        self.init_val = 0.0
        self.value = self.init_val

    # Init_val can be any object for which scalar multiplication, scalar division, and += are defined
    def initialize(self, init_val, count_init_val=None):
        self.init_val = init_val
        if count_init_val is not None:
            self.count_init_val = count_init_val
        self.reset()

    def update(self, value, num=1):
        self.count += num
        self.value += value #Not sure if that works for all losses, but works for accuracy

    def reset(self):
        self.count = self.count_init_val
        self.value = self.init_val

    def average(self):
#        if self.count > 0:
#            return self.value / self.count
#        else:
#            return 0
        try:
            returnval = self.value / self.count
        except ZeroDivisionError:
            returnval = self.init_val

        return returnval


class Meters:
    def __init__(self, *args):
        self.meters = {}
        for name in args:
            self.add_meter(name)

    def add_meter(self, name):
        self.meters[name] = AverageMeter()

    def initialize_meter(self, name, init_val, count_init_val=None):
        self.meters[name].initialize(init_val, count_init_val)

    def reset_meter(self, name):
        self.meters[name].reset()

    def average(self, name):
        return self.meters[name].average()

    def update(self, name, value, num=1):
        self.meters[name].update(value, num)

    def reset_all(self):
        for k in self.meters:
            self.reset_meter(k)
