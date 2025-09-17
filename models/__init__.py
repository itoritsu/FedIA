import os
import importlib

def get_all_models():
    current_dir = os.path.dirname(__file__)
    return [
        fname.split('.')[0]
        for fname in os.listdir(current_dir)
        if fname.endswith('.py') and '__' not in fname
    ]

names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower():x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)

def get_model(nets_list,args, transform):
    return names[args.model](nets_list,args,transform)
