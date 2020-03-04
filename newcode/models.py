# Models

import torch
import torch.nn as nn


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
    # Make these depend on specific model params, for exactness
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
        filtered_out = tuple(out[k] for k in self.return_keys)
        if len(filtered_out) == 1:
            filtered_out = filtered_out[0]
        return filtered_out

    # def get_params()
    #  overload this if needed

'''
class Tracker:
'''

# Collect several models and relate them
# Track the models, their optimizers, and the loss criteria
class ModelCollection:
    def __init__(self, **kwargs):
        self.models = kwargs.get('models',[])
        self.model_folders = kwargs.get('model_paths',[])
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
    def collection_name(self):
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
    def log(self, stats):
        pass #use tensorboard

    # Save/load interface? Yes, need to have this
    def save_collection(self, checkpoint_num=None):
        assert(len(models) == len(model_paths))
        timestr = timestamp_string()
        append_num = str(checkpoint_num) if (checkpoint_num is not None) else ''
        for i, m in enumerate(self.models):
            m.save_model(os.path.join(self.model_folders[i], 
                                      self.collection_name, 
                                      m.model_instance_name+'_'+m.model_class_name+'_'+append_num+'.pth'))


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
            model_collection.log(batch_stats) #Implement this
            if i%print_every == 0:
                print(batch_stats)
        summary = model_collection.tracker_summary()
        model_collection.log(summary)
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

