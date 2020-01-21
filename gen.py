"""
This script generates the HDF5 datafile that will be used as the source of the classification data tasks. It requires
that the L2DATA environment variable be set before running this script (see classification_tasks README), which
specifies the top-level folder under which to look for the pre-generated data.
"""

import os
import random

import numpy as np
from learnkit.data_util import generate_data, DataGenerationModelInterface
from learnkit.utils import module_relative_file


class DummyArcadeModel(DataGenerationModelInterface):
    """Dummy model to interact with l2arcade game during data generation. More intelligent models can be provided to
    generate data with more intelligent behaviors."""
    def step(self, state):
        action = random.choice([None, 1, 2, 3, 4])
        return action

    def update(self, reward):
        pass


def run_arcade_episode(episode):
    """
    Run an the episode as an RL task, collecting state, reward, and action data and aggregating it to be returned
         in the informational dictionaries. The output will be saved to an HDF5 file and loaded later as classification
         task data.
    :param episode: (obj) l2arcade episode object, which behaves like an RL environment
    :return: (dict, dict, dict) Three dictionaries of classification task data. This currently includes:
        task_dict: Task level information. Required key: 'name'
        params_dict: Task parametrization level data, specific to the task with a set of parameters: Required key:
            'parameter_values', which are the specific parameter values for that task.
        episode_dict: Episode level information, which is the main data used for classification. Required keys:
            'states': lx128x128x3 numpy array corresponding to the frames of the l2arcade game, where l=length of the
                episode
            'rewards' lx1 numpy array of rewards for each step of the episode; l=length of the episode
            'actions' lx1 numpy array of actions taken by agent at each step of the episode; l=length of the episode
    """
    done = False
    model = DummyArcadeModel()
    max_episode_steps = 100
    state = episode.reset()
    steps, max_steps = 0, max_episode_steps
    states_arr = np.expand_dims(state, 0)
    rewards = [0]
    actions = [None]
    while steps < max_steps and not done:
        action = model.step(state)
        actions.append(action)
        state, reward, done, info = episode.step(action)
        states_arr = np.concatenate((states_arr, np.expand_dims(state, 0)), 0)
        rewards.append(reward)
        model.update(reward)
        steps += 1

    # NOTE: episode._task is None until episode.reset is called.
    task_dict = {'name': episode.info.name}
    params_dict = {'parameter_values': episode._task.parameter_values()[0]}
    episode_dict = {'states': states_arr, 'rewards': np.array(rewards), 'actions': np.array(actions, dtype=np.float)}
    return task_dict, params_dict, episode_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample data generator for classification data.")
    parser.add_argument('--syllabus', default=module_relative_file(__file__, "gen_syllabus"))
    args = parser.parse_args()
    syllabus_path = os.path.splitext(args.syllabus)[0]
    # 'generate_data' uses the name of the syllabus without the extension, which is currently '.json'
    syllabus_basename = os.path.basename(syllabus_path)

    # 'generate_data' reads the syllabus and calls 'run_arcade_episode' for each episode in the syllabus
    generate_data(syllabus_path, run_arcade_episode, syllabus_seed=1235)

