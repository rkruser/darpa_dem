# PyTorch dataloader for the generated h5py data

from dataset import *

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.nn.functional import interpolate

L2DATA = os.getenv('L2DATA')


############################################################################
# Pong batches and dataset objects
############################################################################

class PongBatch(DataBatchBase):
    keys = ['state', 'outcome', 'bg_color', 'paddle_size']

    @staticmethod
    def data_keys():
        return PongBatch.keys

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PongDataset(torch.utils.data.Dataset):
    def __init__(self, size, **kwargs):
        super().__init__() #?
        self.keys = PongBatch.data_keys()
        for k in self.keys:
            setattr(self, k, kwargs[k])
        self.size = size 

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        kwargs = {k:getattr(self,k)[index].unsqueeze(0) for k in self.keys}
        return PongBatch(**kwargs)

############################################################################
# Dataset construction functions
############################################################################

# Discretize color
def standard_color_map(rgb_triple):
    if rgb_triple[0] > 128:
        return 1
    else:
        return 0

# Discretize reward
def standard_reward_map(reward_value):
    if reward_value <= 0:
        return 0
    else:
        return 1

# Large = 1, small = 0 # For weird code history reasons
def standard_size_map(size):
    if size > 0.2:
        return 1
    else:
        return 0

def identity_map(size):
    return size

def Construct_Pong_Dataset(json_file, 
                 train_proportion=0.8, 
                 color_map = standard_color_map, 
                 reward_map = standard_reward_map, 
                 size_map = standard_size_map,
                 game_keys=['l2arcadekit.l2agames:Pong'],
                 samples_per_game = None,
                 resize=None,
                 noise=None,
                 cutoff=None #float for where to cut off the dataset
                 ):
    name = os.path.basename(os.path.splitext(json_file)[0])
    fullpath = os.path.join(L2DATA, 'data', name, name+'.hdf5')
#    color_map = color_map # How to turn rgb colors into discrete labels
#    reward_map = reward_map
#    size_map = size_map
#    game_keys = game_keys
    
    datafile = h5py.File(fullpath, 'r')
    counts = []
    all_data = {'state':[], 'outcome':[], 'bg_color':[], 'paddle_size':[]}
    for game_key in iter(datafile):
        if game_keys is not None:
            if game_key not in game_keys:
                continue
        game = datafile[game_key]
        
        for param_hash in iter(game):
            param_set = game[param_hash]
            param_values = json.loads(param_set.attrs['parameter_values'])
            current_color = torch.Tensor(1).fill_(color_map(param_values['bg_color']))

            # No size map here?
            current_size_agent = torch.Tensor(1).fill_(size_map(param_values['agent/paddle/width']))
#            current_size_bot = torch.Tensor(1).fill_(size_map(param_values['bot/paddle/width']))
#             Note difference between bot/paddle/width and agent/paddle/width!
            
            for episode_id in iter(param_set):
                episode = param_set[episode_id]
            
                rewards_dset = episode['rewards']
                states_dset = episode['states']
            
                rewards = np.zeros(rewards_dset.shape)
                states = np.zeros(states_dset.shape)
            
                rewards_dset.read_direct(rewards)
                states_dset.read_direct(states)
            
                rewards = torch.Tensor(rewards)
                reward_sum = rewards.sum().item() #Must get before cutoff for final score

                states = torch.Tensor(states)
                states = (states-states.min())/(states.max()-states.min())
                # Make states between 0 and 1

#                states = torch.Tensor(states) / 255.0 # what are the values between anyway?
#                states = torch.Tensor(states).permute(0,3,1,2) / 255.0
                # Apparently this is in RBG, not RGB, need to rearrange here or in files
#                color_perm = torch.LongTensor([0,2,1]) # Undo RBG
#                states = states[:,color_perm, :, :]

                # If we want to chop off the last part of a game
                if cutoff is not None:
                    cutoff_index = int(cutoff*len(states))
                    states = states[:cutoff_index]

                # No bias for long games
                if samples_per_game is not None:
                    sample_inds = torch.randint(len(states), [samples_per_game])
                    states = states[sample_inds]

                if resize is not None:
                    states = interpolate(states, size=resize, mode='bilinear')
                if noise is not None:
                    states = states+noise*torch.randn(states.size())
                    #states = (states-states.min())/(states.max()-states.min())

                all_data['state'].append(states)
                all_data['outcome'].append(torch.Tensor(len(states)).fill_(reward_map(reward_sum)))
                all_data['bg_color'].append(current_color.repeat(len(states)))
                all_data['paddle_size'].append(current_size_agent.repeat(len(states)))
                counts.append(len(states))

    datafile.close()

    size = int(np.sum(counts)) #need int here for pytorch sake; can't handle np.int64
    all_data = {k:torch.cat(all_data[k]) for k in all_data}

    train_size = int(train_proportion*size)
    test_size = size - train_size
    inds = torch.randperm(size)
    train_inds = inds[:train_size]
    test_inds = inds[train_size:]

    train_data = {k:all_data[k][train_inds] for k in all_data}
    test_data = {k:all_data[k][test_inds] for k in all_data}
    
    train_set = PongDataset(train_size, **train_data)
    test_set = PongDataset(test_size, **test_data)

    return train_set, test_set


