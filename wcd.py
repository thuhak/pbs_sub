#!/usr/bin/env python3
"""
show target directory
"""
from argparse import ArgumentParser
from HPCSub.config import HPCFile


parser = ArgumentParser(description=__doc__)
parser.add_argument('path', help='HPC file path')
args = parser.parse_args()
path = HPCFile(args.path)

if path.is_dir():
    print(path)
elif path.is_file():
    print(str(path.parent))
