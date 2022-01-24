"""
load all software parser
"""
import importlib
import os
from .mainparser import MainParser

utils = {'database', 'mainparser'}
for module in os.listdir(os.path.dirname(__file__)):
    if module.endswith('.py'):
        name = os.path.basename(module).rsplit('.', maxsplit=1)[0]
        if name not in utils:
            importlib.import_module(f'.{name}', __package__)
