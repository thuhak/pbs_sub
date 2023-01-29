#!/usr/bin/env python3
# author: thuhak.zhou@nio.com
import os
from HPCSub import SubParser


if __name__ == '__main__':
    test = True if os.environ.get('SUB_TEST') else False
    parser = SubParser()
    result = parser.run(test=test)
