# Project interface prototype
'''
In this file I will template out an interface for running experiments.



Hyperparams:
- Use kwargs in model + dictionary of params
- Have a function that takes the dictionary and renders command line arguments to change the values?
- 





'''


# System
import sys
import argparse
import json
import os

# Pytorch
import torch
import torch.nn as nn

# Take one or more dictionaries and parse their options from the command line
# Alternatively, take a config file from --config and parse the options that way
# Return the same number of dictionaries as fed in
def parse_dict_options(*arg_dicts, arg_list=sys.argv[1:], arg_descriptions={}):
    class bool_reader:
        def __init__(self, s):
            if isinstance(s, str):
                self.state = s.lower() in {'true', 't', 'y', '1'}
            else:
                self.state = bool(s)


    arg_dict = {}
    for d in arg_dicts:
        arg_dict.update(d)

    parser = argparse.ArgumentParser()
    for arg in arg_dict:
        val = arg_dict[arg]
        if isinstance(val, bool):
            parser.add_argument('--'+str(arg), type=bool_reader, default=bool_reader(val), 
                               help=arg_descriptions.get(arg,''))
        else:
            parser.add_argument('--'+str(arg), type=type(val), default=val, 
                                help=arg_descriptions.get(arg,''))

    parser.add_argument('--config', type=str, default='', help='Config json file to load from.')
    parse_dict = vars(parser.parse_args(arg_list))
    new_dict = arg_dict

    configname = parse_dict['config']
    if os.path.isfile(configname):
        print("Using the config file {0}".format(configname))
        with open(configname, 'r') as fhandle:
            config_dict = json.load(fhandle)
        new_dict.update(config_dict)
    else:
        for arg in parse_dict:
            val = parse_dict[arg]
            if isinstance(val, bool_reader):
                new_dict[arg] = val.state
            else:
                new_dict[arg] = val

    new_dicts = tuple({k:new_dict[k] for k in new_dict if k in d} for d in arg_dicts)
    if len(new_dicts) == 1:
        new_dicts = new_dicts[0]

    return new_dicts


'''
def move_to_device(*args):
    for a in args:
'''      


# Inherit from nn.module?
# Don't bother with data_parallel at the moment
# Doesn't need to reference cuda explicitly
class Model(nn.Module):

    # Return a dict of default params and values
    @staticmethod
    def default_params_dict():
        pass

    def __init__(self, **kwargs):
        super().__init__()
        defaults = self.__class__.default_params_dict()
        for d in defaults:
            setattr(self, d, defaults[d])
        for k in kwargs:
            setattr(self, k, kwargs[k]) # can this be called from self?

    # Return model name
    @property
    def model_class_name(self):
        pass

    @property
    def model_instance_name(self):
        pass

    # Ordered list of keys for layer outputs
    @property
    def return_keys(self):
        pass

    def save_model(self, location):
        torch.save(self.state_dict(), location)

    def load_model(self, location):
        self.load_state_dict(torch.load(location))

    # Return a dict of intermediates
    def _forward(self, x):
        pass

    def forward(self, x):
        out = self._forward(x)
        filtered_out = tuple(out[k] for k in self.returnkeys)
        if len(filtered_out) == 1:
            filtered_out = filtered_out[0]
        return filtered_out

    # def get_params()
    #  overload this if needed



# Collect several models and relate them
# Track the models, their optimizers, and the loss criteria
class ModelCollection:
    def __init__(self, **kwargs):
        self.models = kwargs.get('models',[])
        self.optimizers = kwargs.get('optimizers',[])
        self.trackers = kwargs.get('trackers', [])
        self.opts = kwargs.get('opts',{})
        self.name = kwargs.get('name', 'model_collection')

        # Eval func takes a unit of data (batch, point, etc.)
        #  and runs the whole collection through to evaluation
        self.evaluate_func = kwargs.get('eval_function',None)
        self.evaluate_and_update_func = kwargs.get('eval_update_function',None)
        self.tracker_summary_func = kwargs.get('tracker_summary_func', None)
#        self.backward_func = kwargs.get('backward_function', None)
#        self.reset_grad_func = kwargs.get('reset_grad_func', None)
#        self.grad_backward_func = kwargs.get('grad_backward_func', None)
#        self.update_trackers_func = kwargs.get('update_trackers_func', None)

#        self.current_state = None

    @property
    def name(self):
        return self.name

    def evaluate_and_update(self, x, **opts):
        return self.evaluate_and_update_func(x, self.models, self.optimizers, self.trackers, **opts)

    def evaluate_only(self, x, **opts):
        return self.evaluate_func(x, self.models, self.trackers, **opts)

    def tracker_summary(self, **opts):
        return self.tracker_summary_func(self.trackers, **opts)

    def end_training_criterion(self, tracker_summary):
        return False

    def reset_trackers(self):
        for t in self.trackers:
            t.reset()

    # Logging interface?


    # Save/load interface?
    def save_collection(self, checkpoint_num=None):
        pass


    # Leave room for inheritance?




# Put cuda info into dataloader
# model_collection already initialized in right way
def train_test_loop(model_collection, dataloader, nepochs=10, checkpoint_every=10, print_every=10, test_mode=False):
    if test_mode:
        nepochs = 1
        #print_every = len(dataloader)+1
        batch_func = lambda batch: model_collection.evaluate(batch)
    else:
        batch_func = lambda batch: model_collection.evaluate_and_update(batch)

    for epoch in range(nepochs):
        for i, batch in enumerate(dataloader):
            batch_stats = batch_func(batch)
            if i%print_every == 0:
                print(batch_stats)
        summary = model_collection.tracker_summary()
        model_collection.reset_trackers()
        print(summary)
        if test_mode:
            loop_end = True
        else:
            loop_end = model_collection.end_training_criterion(summary)
        if ((not test_mode) and (loop_end or epoch%checkpoint_every==0)):
            model_collection.save_collection(checkpoint_num=epoch)
        if loop_end:
            break
        



### Now for dataloader abstractions

# Generally set pin_memory to True for gpu use,
#  and overload pin_memory function in data return

class DataLoader_To_Device(torch.utils.data.DataLoader):
    def __init__(self, device, **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def __iter__(self):
        for v in super().__iter__():
            yield v.to(self.device)

# Return a dataloader
# See https://pytorch.org/docs/stable/data.html
def construct_data_loader(dataset, **kwargs):
    loader_args = {
        'device':'cpu',
        'batch_size':64,
        'collate_fn':None,
        'num_workers':4
    }
    loader_args.update(kwargs)
    return torch.utils.data.DataLoader(dataset, **loader_args)


'''
    def cuda(self):
        for i in len(self.models):
            self.models[i] = self.models[i].cuda() # Will this change the object in-place?

    def to(self, device):
        for i in len(self.models):
            self.models[i] = self.models[i].to(device)

    def reset_grad(self):
        if self.reset_grad_func:
            self.reset_grad_func(self.models)
        else:
            for m in self.models:
                m.zero_grad()

    def grad_backward(self):
        if self.grad_backward_func:
            self.grad_backward_func(self.current_state)
        else:
            for 

    # Run the optimizers
    def update(self):
        for opt in self.optimizers:
            opt.step()
'''


# Define methods to save the model, checkpoint, get params, etc.




def test1():
    d = {'a':9, 'b':'six', 'cee':22, 'gee':True}
    z = {'neep': 78}
    args = ['--a','20', '--b', 'sdfkj', '--gee', 'False']
    desc = {'a':'heeeey'}
    newd, newz = parse_dict_options(d, z, arg_descriptions=desc)
    print(newd, newz)



if __name__=='__main__':
    test1()
