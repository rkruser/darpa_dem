# PyTorch dataloader for the generated h5py data

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.nn.functional import interpolate

L2DATA = os.getenv('L2DATA')


# Red = 1, blue = 0

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

# Large = 1, small = 0
def standard_size_map(size):
    if size > 0.2:
        return 1
    else:
        return 0


def Construct_L2M_Dataset(json_file, train_proportion=0.8, color_map = standard_color_map, 
                 reward_map = standard_reward_map, size_map = standard_size_map,
                 game_keys=['l2arcadekit.l2agames:Pong'],
                 correlate_exactly = False,
                 resize=None,
                 noise=None
                 ):
    name = os.path.basename(os.path.splitext(json_file)[0])
    fullpath = os.path.join(L2DATA, 'data', name, name+'.hdf5')
#    color_map = color_map # How to turn rgb colors into discrete labels
#    reward_map = reward_map
#    size_map = size_map
#    game_keys = game_keys
    
    datafile = h5py.File(fullpath, 'r')
    all_states = []
    all_actions = []
    all_rewards = []
    counts = []
    all_labels = {'reward':[], 'bg_color':[], 'bot/paddle/width':[]}
    for game_key in iter(datafile):
        if game_keys is not None:
            if game_key not in game_keys:
                continue
        game = datafile[game_key]
        
        for param_hash in iter(game):
            param_set = game[param_hash]
            param_values = json.loads(param_set.attrs['parameter_values'])
            current_color = torch.Tensor(1).fill_(color_map(param_values['bg_color']))
            current_size = torch.Tensor(1).fill_(size_map(param_values['bot/paddle/width']))
#            print(current_color, current_size)
            
            for episode_id in iter(param_set):
                episode = param_set[episode_id]
            
                rewards_dset = episode['rewards']
                actions_dset = episode['actions']
                states_dset = episode['states']
            
                rewards = np.zeros(rewards_dset.shape)
                actions = np.zeros(actions_dset.shape)
                states = np.zeros(states_dset.shape)
            
                rewards_dset.read_direct(rewards)
                actions_dset.read_direct(actions)
                states_dset.read_direct(states)
            
                rewards = torch.Tensor(rewards)
                states = torch.Tensor(states).permute(0,3,1,2) / 255.0
                # Apparently this is in RBG, not RGB, need to rearrange here or in files
#                color_perm = torch.LongTensor([0,2,1]) # Undo RBG
#                states = states[:,color_perm, :, :]
                if resize is not None:
                    states = interpolate(states, size=resize, mode='bilinear')
                if noise is not None:
                    states = states+noise*torch.randn(states.size())
                    #states = (states-states.min())/(states.max()-states.min())
                actions = torch.Tensor(actions)
            
                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                counts.append(len(states))
                all_labels['reward'].append(torch.Tensor(len(states)).fill_(reward_map(rewards.sum().item())))
                all_labels['bg_color'].append(current_color.repeat(len(states)))
                all_labels['bot/paddle/width'].append(current_size.repeat(len(states)))

    datafile.close()

    size = int(np.sum(counts)) #need int here for pytorch sake; can't handle np.int64
    states = torch.cat(all_states)
    rewards = torch.cat(all_rewards)
    actions = torch.cat(all_actions)
    labels = {'reward':torch.cat(all_labels['reward']),
              'bg_color':torch.cat(all_labels['bg_color']),
              'bot/paddle/width':torch.cat(all_labels['bot/paddle/width'])}

#    print("Overall dataset", stat_grid(labels['reward'], 
#            labels['bot/paddle/width'], labels['bg_color']))

    train_size = int(train_proportion*size)
    test_size = size - train_size
    inds = torch.randperm(size)
    train_inds = inds[:train_size]
    test_inds = inds[train_size:]
    train_set = L2M_Pytorch_Dataset(train_size, states[train_inds], 
                                    rewards[train_inds], actions[train_inds],
                                    {'reward':labels['reward'][train_inds],
                                     'bg_color':labels['bg_color'][train_inds],
                                     'bot/paddle/width':labels['bot/paddle/width'][train_inds]})
#                                     noise=noise)
    test_set = L2M_Pytorch_Dataset(test_size, states[test_inds], 
                                    rewards[test_inds], actions[test_inds],
                                    {'reward':labels['reward'][test_inds],
                                     'bg_color':labels['bg_color'][test_inds],
                                     'bot/paddle/width':labels['bot/paddle/width'][test_inds]})
#                                     noise=noise)

    print("\nTrain stats")
    train_set.print_statistics()

    print("\nTest stats")
    test_set.print_statistics()
       

    return train_set, test_set


# Create a grid of average value given property
# xproperty and yproperty are [0/1] tensors
def stat_grid(values, xproperty, yproperty):
    nx_ny = (xproperty<=0.5)&(yproperty<=0.5)
    nx_yy = (xproperty<=0.5)&(yproperty>0.5)
    yx_ny = (xproperty>0.5)&(yproperty<=0.5)
    yx_yy = (xproperty>0.5)&(yproperty>0.5)
    val_nx_ny = values[nx_ny]
    val_nx_yy = values[nx_yy]
    val_yx_ny = values[yx_ny]
    val_yx_yy = values[yx_yy]

    sumgrid = torch.Tensor([[val_nx_ny.sum(), val_nx_yy.sum()], [val_yx_ny.sum(), val_yx_yy.sum()]])
    totalgrid = torch.Tensor([[len(val_nx_ny), len(val_nx_yy)], [len(val_yx_ny), len(val_yx_yy)]])
    proportiongrid = sumgrid/totalgrid
    return sumgrid, totalgrid, proportiongrid

def print_grid(grid, xlabels, ylabels):
    print('    ', end='')
    for j in range(len(ylabels)):
        print(ylabels[j]+' ', end='')
    print('\n', end='')
    for i in range(len(grid)):
        print(xlabels[i]+'  ', grid[i])

class L2M_Pytorch_Dataset(Dataset):
    def __init__(self, size, states, rewards, actions, labels): #,noise=None):
        self.size = size 
        self.states = states 
        self.rewards = rewards 
        self.actions = actions
        self.labels = labels
#        self.noise = noise

    def __len__(self):
        return self.size


    def __getitem__(self, index):
#        return self.states[index], self.labels['reward'][index], self.labels['bg_color'][index], self.labels['bot/paddle/width'][index]
        labels = { 'reward': self.labels['reward'][index], 
                                    'bg_color': self.labels['bg_color'][index], 
                                    'bot/paddle/width': self.labels['bot/paddle/width'][index] }
#        if self.noise is None:
        return self.states[index], labels
#        else:
#            im = self.states[index]
#            return im+self.noise*torch.randn(im.size()), labels

    def statistics(self):
        return stat_grid(self.labels['reward'], self.labels['bot/paddle/width'], self.labels['bg_color'])

    def print_statistics(self):
        counts, totals, proportions = self.statistics()
        print("Counts")
        print_grid(counts, ["Small", "Large"], ["Blue", "Red"])
        print("Totals")
        print_grid(totals, ["Small", "Large"], ["Blue", "Red"])
        print("Proportions")
        print_grid(proportions, ["Small", "Large"], ["Blue", "Red"])





