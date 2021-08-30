import json
import torch
from collections import OrderedDict

class ConfigParser:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.config = json.load(f, object_hook=OrderedDict)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = dict(dict(self.config[name])['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, item):
        return self.config[item]

if __name__ == "__main__":
    config = ConfigParser("./config.json")
    # model = config.init_obj('model', module)
    # model.eval()
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # criterion = config.init_obj("optimizer", torch.optim, trainable_params)
    # print(model)
    pass