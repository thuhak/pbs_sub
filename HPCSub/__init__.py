"""
load all app parser
"""
# author: thuhak.zhou@nio.com
import importlib
import os
import types

from .mainparser import SubParser
from .config import *

for app in all_app_config.keys():
    default_cls = os.path.join(script_root, 'apps', f'{app}.py')
    if os.path.isfile(default_cls):
        importlib.import_module(f'.apps.{app}', __package__)
    else:
        types.new_class(app.capitalize(), (SubParser,), {})
