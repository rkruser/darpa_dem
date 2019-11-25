# PyTorch dataloader for the generated h5py data

import os
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils


L2DATA = os.getenv('L2DATA')

# Discretize color
def standard_color_map(rgb_triple):
    if rgb_triple[0] > 128:
        return 0
    else:
        return 1

# Discretize reward
def standard_reward_map(reward_value):
    if reward_value <= 0:
        return 0
    else:
        return 1

class L2M_Pytorch_Dataset(Dataset):
    def __init__(self, experiment_json, color_map = standard_color_map, 
                 reward_map = standard_reward_map, game_keys=['l2arcadekit.l2agames:Pong']):
        self.json_file = experiment_json
        self.name = os.path.basename(os.path.splitext(self.json_file)[0])
        self.fullpath = os.path.join(L2DATA, 'data', self.name, self.name+'.hdf5')
        self.color_map = color_map # How to turn rgb colors into discrete labels
        self.reward_map = reward_map
        self.game_keys = game_keys

        self.file = h5py.File(self.fullpath, 'r')
        all_states = []
        all_actions = []
        all_rewards = []
        counts = []
        all_labels = {'reward':[], 'bg_color':[]}
        for game_key in iter(self.file):
            if game_keys is not None:
                if game_key not in game_keys:
                    continue
            game = self.file[game_key]

            for param_hash in iter(game):
                param_set = game[param_hash]
                param_values = json.loads(param_set.attrs['parameter_values'])
                current_color = torch.Tensor(1).fill_(self.color_map(param_values['bg_color']))

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
                    actions = torch.Tensor(actions)

                    all_states.append(states)
                    all_actions.append(actions)
                    all_rewards.append(rewards)
                    counts.append(len(states))
                    all_labels['reward'].append(torch.Tensor(len(states)).fill_(self.reward_map(rewards.sum().item())))
                    all_labels['bg_color'].append(current_color.repeat(len(states)))


        self.size = np.sum(counts)
        self.states = torch.cat(all_states)
        self.rewards = torch.cat(all_rewards)
        self.actions = torch.cat(all_actions)
        self.labels = {'reward':torch.cat(all_labels['reward']),
                       'bg_color':torch.cat(all_labels['bg_color'])}
        self.file.close()


    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return self.states[index], self.labels['reward'][index], self.labels['bg_color'][index]
                    


















