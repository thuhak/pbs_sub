"""
configuration and global variable
"""
# author: thuhak.zhou@nio.com
import os
import time
import logging
import subprocess
from enum import Enum
from pathlib import Path
from pwd import getpwuid
from typing import Optional, List, Dict

import toml
import requests
from pydantic import BaseModel

__all__ = ["conf", "all_app_config", "AppType", "AppModel", "HPCFile", "script_root", "logger", "stream_log", "user",
           "groups", "email", "api", "local"]
user = getpwuid(os.getuid())[0]
groups = frozenset(subprocess.getoutput('groups').split())

logger = logging.getLogger('sub')
logger.setLevel(logging.DEBUG)
log_fmt = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_log = logging.StreamHandler()
stream_log.setFormatter(log_fmt)
logger.addHandler(stream_log)
if os.path.isdir(log_dir := '/var/log/pbs_sub'):
    file_log = logging.FileHandler(os.path.join(log_dir, user) + '.log')
    file_log.setFormatter(log_fmt)
    file_log.setLevel(logging.DEBUG)
    logger.addHandler(file_log)

script_root = os.path.dirname(__file__)

with open('/etc/sub.toml') as f:
    conf = toml.load(f)

email = os.environ.get("SUB_MAIL") or f'{user}@{conf["mail_domain"]}'
local = conf.get('location')
default_smb = conf.get('samba_host', '')
local_file_root = conf.get('local_file_root', '')


class Storage(BaseModel):
    Base: Path
    Level: int


class HPCFile:
    _storages = [Storage(**v) for v in conf['storage']]

    def __init__(self, path: str):
        suffix = os.environ.get("SUB_FILE_ROOT") or default_smb
        if suffix and path.lower().startswith(suffix.lower()):
            path = path[len(suffix):]
            if root := (os.environ.get('SUB_HPC_ROOT') or local_file_root):
                path = root + path
            logger.debug(f'change path to {path}')
        p = Path(path.replace('\\', '/'))
        parents = p.resolve().parents
        level = 0
        for storage in self._storages:
            if storage.Base in parents:
                level = storage.Level
                break
        self.level = level
        self._path = p.absolute()

    def __getattr__(self, item):
        if item != '_path':
            return getattr(self._path, item)

    def __str__(self):
        return str(self._path)

    def __fspath__(self):
        return str(self._path)

    def __repr__(self):
        return str(self._path)

    def is_executable(self):
        return os.access(self._path, os.X_OK)


class AppType(Enum):
    Simulation = 'simulation'
    PreProcessing = 'pre-processing'
    PostProcessing = 'post-processing'
    UserScript = 'user-script'


class AppModel(BaseModel):
    """
    HPC software model
    """
    Name: str
    Type: AppType = AppType.Simulation
    DefaultVersion: Optional[str] = None
    Versions: List[str] = []
    DefaultMinCores: int = 0
    TinyJob: bool = False
    MaxCores: int = 0
    MPI: Optional[bool] = None
    OpenMP: int = 0
    MaxGPU: int = 0
    DefaultGPU: int = 0
    DefaultCoreWithGPU: int = -1
    Memory: int = 0
    LicenseCost: int = 0
    LicenseName: Optional[str] = None
    Formats: List[str] = []
    Description: Optional[str] = None

    def __hash__(self):
        return id(self)


# loading app config
all_app_config: Dict[str, AppModel] = {}
for c in Path(os.path.join(script_root, 'templates')).glob('*.toml'):
    with open(c) as f:
        v = toml.load(f)
    name = v['Name']
    all_app_config[name] = AppModel(**v)


class HPCWebAPI:
    def __init__(self, base_url, api_user, api_pass, api_location):
        self.session = requests.Session()
        self.session.verify = False
        self.session.auth = (api_user, api_pass)
        self.sites = self.session.get(f'{base_url}/pbs').json()['site']
        if api_location is None:
            if os.path.exists(site_cache := '/dev/shm/site'):
                with open(site_cache) as f:
                    api_location = f.read()
            else:
                with open('/etc/pbs.conf') as f:
                    for l in f:
                        if l.startswith('PBS_SERVER'):
                            local_server = l.split('=')[1].strip()
                            break
                api_location = self.get_location(local_server)
                with open(site_cache, 'w') as f:
                    f.write(api_location)
                os.chmod(site_cache, 0o666)
        self.location = api_location
        self.base_url = f'{base_url}/pbs'

    def delay(self, location=None):
        site_info = self.session.get(f'{self.base_url}/{location or self.location}').json()
        return int(time.time()) - site_info['timestamp']

    def get_location(self, site):
        try:
            site_loc = [s['location'] for s in self.sites if s['name'] == site][0]
        except IndexError:
            logger.error(f"invalid site {site}")
            exit(1)
        return site_loc

    def info(self, sub_url, location=None, **kwargs):
        return self.session.get(f'{self.base_url}/{location or self.location}/{sub_url}', params=kwargs).json()


api_conf = conf['api']
api = HPCWebAPI(api_conf['url'], api_conf['user'], api_conf['passwd'], api_location=local)
