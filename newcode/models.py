# Specific models

from model_base import *
from data_base import DataBatchBase

###############################################
# Net builder functions
###############################################

def get_conv_net():
    pass

def get_linear_net():
    pass


###############################################
# Network models
###############################################

class Conv_Net_Classifier(Model):
    @staticmethod
    def default_params_dict():
        pdict = {
            'name':'conv_net_classifier_inst',
            'keys':['embedding','prediction']
        }
        return pdict

    @classmethod
    def model_class_name(cls):
        return str(cls)

    @property
    def model_instance_name(self):
        return self.name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_network = get_conv_net()
        self.classifier_network = get_linear_net()

    @property
    def return_keys(self):
        return self.__class__.default_params_dict()['keys']


    class Conv_Net_Classifier_ReturnType(DataBatchBase):
        @staticmethod
        def data_keys():
            return Conv_Net_Classifier.default_params_dict()['keys']
            

    def forward(self, x):
        emb = self.embedding_network(x)
        pred = self.classifier_network(emb)
        return Conv_Net_Classifier_ReturnType(embedding=emb, prediction=pred)
        


class Linear_Classifier(Model):
    @staticmethod
    def default_params_dict():
        pdict = {
            'name':'linear_classifier_inst',
            'keys':['prediction']
        }
        return pdict

    @classmethod
    def model_class_name(cls):
        return str(cls)

    @property
    def model_instance_name(self):
        return self.name

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier_network = get_linear_net()

    @property
    def return_keys(self):
        return self.__class__.default_params_dict()['keys']

    class Linear_Net_Classifier_ReturnType(DataBatchBase):
        @staticmethod
        def data_keys():
            return Conv_Net_Classifier.default_params_dict()['keys']
            
    def forward(self, x):
        pred = self.classifier_network(x)
        return Conv_Net_Classifier_ReturnType(prediction=pred)
   

###############################################
# Model Collections
###############################################

# Assumes one data batch of DataBatchBase type with the right keys
# Assume model_list corresponds to optimizer_list

# Right now this is pseudocode
def eval_and_update_adversarial_model(data_batch, models, optimizers, criteria, trackers, **kwargs):
    embedding, outcome_prediction = models.outcome_predictor(data_batch.state)
    color_prediction = models.color_predictor(embedding) #detach or not?

    outcome_predictor_loss = criteria.outcome_predictor_loss(outcome_prediction, data_batch.outcome)
    color_predictor_loss = criteria.color_predictor_loss(color_prediction,data_batch.bg_color)

    outcome_accuracy = criteria.accuracy(outcome_prediction, data_batch.outcome)
    color_accuracy = criteria.accuracy(color_prediction, data_batch.bg_color)

    # Deal with trackers later
    # trackers.sfdlkj

    models.outcome_predictor.zero_grad()
    models.color_predictor.zero_grad()

    outcome_predictor_loss.backward()
    color_predictor_loss.backward()

    optimizers.outcome_predictor_opt.step()
    optimizers.color_predictor_opt.step()

    return trackers.meter.summary
    

# Same as above without the gradient stuff
def eval_adversarial_model(data_batch, models, criteria, trackers, **kwargs):
    pass

def tracker_summary_func():
    pass


# Return an object of type ModelCollection
def construct_adversarial_model():
    models = {
        'outcome_predictor':None,
        'color_predictor':None
    }

    optimizers = {

    }

    criteria = {

    }

    trackers = {

    }




