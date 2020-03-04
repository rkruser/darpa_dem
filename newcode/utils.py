# utils
import sys
import argparse
import json
import os
import time




def timestamp_string():
    return time.strftime('%H:%M:%S_%m_%d_%Y',time.gmtime())

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

# Easy interface for accessing dicts
# basically easydict
class AttrDict:
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])


# Need a function for nice formatted printing of dataset statistics

def print_statistics():
    pass







