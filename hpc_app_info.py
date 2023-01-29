#!/usr/bin/env python
"""
show HPC application configuration
"""
import json
from enum import Enum
from argparse import ArgumentParser

from HPCSub.config import all_app_config

parser = ArgumentParser(description=__doc__)
parser.add_argument('-s', '--software', help='app')


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


if __name__ == '__main__':
    args = parser.parse_args()
    if app := args.software:
        raw = all_app_config.get(app)
        if not raw:
            data = []
        else:
            data = [raw.dict()]
    else:
        data = [x.dict() for x in all_app_config.values()]
    result = {'Apps': data}
    print(json.dumps(result, indent=True, cls=MyEncoder))
