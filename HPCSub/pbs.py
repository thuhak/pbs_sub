"""
object-pbs mapping
"""
# author: thuhak.zhou@nio.com
import os
import re
import json
import subprocess
from enum import Enum
from math import ceil
from subprocess import getstatusoutput
from dataclasses import dataclass, field
from collections import Counter, UserDict
from typing import List, Tuple, Optional, Dict, Union

try:
    from functools import cache
except ImportError:  # py3.8
    from functools import lru_cache as cache

import jmespath
import flashtext
from colorama import Fore, Style
from pydantic import BaseModel
from terminaltables import AsciiTable

from .config import *
from .utils import *

__all__ = ["PbsRes", "Queue", "Placement", "Select", "Queues", "Qsub", "SubError", "Job", "Jobs", "JobState",
           "ServerInfo", "WallTime", "ResourceGroup", "Sharing", "Arrangement"]

is_admin = user in conf['admins']


# utils
class SubError(Exception):
    def __init__(self, err_msg, *args):
        self.err_msg = err_msg
        super(SubError, self).__init__(*args)


# pbs objects
class PbsRes:
    """
    custom pbs resource
    """
    __slots__ = ['value']

    def __init__(self, value):
        self.value = value


class JobState(Enum):
    BatchRunning = 'B'
    Exiting = 'E'
    Finished = 'F'
    Held = 'H'
    Moved = 'M'
    Queued = 'Q'
    Running = 'R'
    Suspended = 'S'
    Transfered = 'T'
    Waiting = 'W'
    UserSuspend = 'U'
    SubJobEnd = 'X'


class Arrangement(Enum):
    Free = 'free'
    Pack = 'pack'
    Scatter = 'scatter'
    VScatter = 'vscatter'

    def __str__(self):
        return str(self.value)


class Sharing(Enum):
    Excl = 'excl'
    Shared = 'shared'
    Exclhost = 'exclhost'

    def __str__(self):
        return str(self.value)


class ResourceGroup(Enum):
    Numa = 'vnode'
    Socket = 'numa'
    Host = 'host'
    IBSwitch = 'ibswitch'

    def __str__(self):
        return f'group={self.value}'


@dataclass
class Placement:
    mapping = {**{x.value: Arrangement for x in Arrangement}, **{x.value: Sharing for x in Sharing}}

    arrangement: Arrangement = Arrangement.Free
    sharing: Optional[Sharing] = None
    group: Optional[ResourceGroup] = None

    def __str__(self):
        return f"place={':'.join(str(x) for x in (self.arrangement, self.sharing, self.group) if x)}"

    @classmethod
    def load(cls, data: str):
        kwargs = {}
        for i in data.split(':'):
            if i in cls.mapping:
                kwargs[(cls.mapping[i]).__name__.lower()] = cls.mapping[i](i)
            else:
                kwargs['group'] = ResourceGroup(i.split('=')[1])
        return cls(**kwargs)


@dataclass
class Select:
    select: int
    ncpus: int = 0
    ngpus: int = 0
    mpiprocs: int = 0
    ompthreads: int = 0
    mem: int = 0

    def __str__(self):
        return ':'.join(f'{k}={int(v)}{"gb" if k == "mem" else ""}' for k, v in vars(self).items() if v)

    @classmethod
    def load(cls, data: str):
        return cls(select=int((items := data.split(':'))[0]),
                   **{(k := i.split('='))[0]: int(re.search(r'\d+', k[1]).group(0)) for i in items[1:]})


@singleton
class ServerInfo:
    """
    HPC server info
    """

    def __init__(self):
        raw_data = api.info('Server')
        self.projects = jmespath.search('data.resources_available.Project', raw_data).split(',')
        self.keyword_parser = flashtext.KeywordProcessor(case_sensitive=False)
        self.keyword_parser.non_word_boundaries.remove('_')
        for p in self.projects:
            self.keyword_parser.add_keyword(p)

    def guess_project(self, path: HPCFile) -> str:
        """
        guess the project name according to the job file path
        """
        projects = self.keyword_parser.extract_keywords(str(path))
        if not projects:
            return 'unknown'
        projects.sort(key=len, reverse=True)
        return projects[0].lower()


@dataclass
class Queue:
    """
    PBS queue model
    """
    Name: str
    Type: str
    Location: str
    CoresPerHost: int
    MemPerHost: int
    CoresPerSocket: int
    CoresPerNuma: int
    UsingCores: int
    WaitingCores: int
    MinCores: int
    MaxCores: int
    OfflineCores: int = 0
    OfflineGPUs: int = 0
    Load: float = 0.0
    GpuPerNuma: int = 0
    RunningJobs: int = 0
    Priority: int = 1
    MaxGPUs: int = 0
    MinGPUs: int = 0
    FreeGPUs: int = 0
    UsingGPUs: int = 0
    WaitingGPUs: int = 0
    Teams: List[str] = field(default_factory=list)
    Apps: List[str] = field(default_factory=list)
    FreeCoresGroup: List[int] = field(default_factory=list)
    FreeGPUsGroup: List[int] = field(default_factory=list)

    def __str__(self):
        return self.Name

    def __hash__(self):
        return id(self)

    def free_core_expr(self, use_gpu=False):
        group = self.FreeGPUsGroup if use_gpu else self.FreeCoresGroup
        return '+'.join(str(k) if v == 1 else f'{k}*{v}' for k, v in
                        Counter(group).items()) if group else '0'

    @property
    def numa_count(self) -> int:
        return self.CoresPerHost // self.CoresPerNuma

    @property
    def socket_count(self) -> int:
        return self.CoresPerHost // self.CoresPerSocket

    @property
    def GpuPerSocket(self):
        return self.GpuPerNuma * self.socket_count

    @property
    def GpuPerHost(self):
        return self.GpuPerNuma * self.numa_count

    @property
    def MemPerCore(self):
        return self.MemPerHost / self.CoresPerHost

    @property
    def MemPerNuma(self):
        return int(self.MemPerCore * self.CoresPerNuma)

    @property
    def has_gpu(self) -> bool:
        return bool(self.GpuPerNuma)

    @property
    def gpu_load(self) -> float:
        try:
            l = round((self.UsingGPUs + self.WaitingGPUs) / (self.UsingGPUs + self.FreeGPUs), 2)
        except ZeroDivisionError:
            l = 0.0
        return l

    def check_app(self, app: str, use_gpu=False):
        if not self.has_gpu and use_gpu:
            raise SubError(f"{self.Name} has no GPU")
        if app not in self.Apps:
            raise SubError(f"{app} can not run at queue {self.Name}")

    @cache
    def recommend_cores(self, min_cores, tiny_job=False):
        """
        recommend size of cores which is greater than min_cores
        """
        if min_cores < -3:
            raise ValueError(f"DefaultMinCores must be greater than -3")
        elif min_cores == -3:
            cores = self.CoresPerNuma
        elif min_cores == -2:
            cores = self.CoresPerSocket
        elif 0 < min_cores <= self.CoresPerNuma:
            if tiny_job:
                cores = min_cores
            else:
                cores = self.CoresPerNuma
        elif self.CoresPerNuma < min_cores <= self.CoresPerSocket:
            cores = min(ceil(min_cores/self.CoresPerNuma) * self.CoresPerNuma, self.CoresPerSocket)
        elif self.CoresPerSocket < min_cores <= self.CoresPerHost or min_cores == -1:
            cores = self.CoresPerHost
        else:
            cores = ceil(min_cores / self.CoresPerHost) * self.CoresPerHost
        return cores

    def priority_score(self, cores, use_gpu=False, tiny_job=False, job_count=1):
        """
        calculate the priority score for specify cores of job
        """
        score = self.Priority
        group = self.FreeGPUsGroup if use_gpu else self.FreeCoresGroup
        load = self.gpu_load if use_gpu else self.Load
        if not use_gpu:
            cores = self.recommend_cores(cores, tiny_job)
        valid_group_count = sum(c // cores for c in group)
        if valid_group_count > job_count:
            score += 100 * job_count
        elif valid_group_count > 0:
            score += 100 * valid_group_count
        else:
            score -= 100 * load
        return score


@singleton
class Queues(UserDict):
    """
    load pbs queue info
    """

    def __init__(self):
        super().__init__()
        self.data = {}
        for q_data in api.info('Queue/*')['data']:
            name = q_data['name']
            cores_per_vnode = jmespath.search('resources_available.ncpu_pernode', q_data)
            cores_per_socket = jmespath.search('resources_available.ncpu_pernuma', q_data) or cores_per_vnode
            q_pri = q_data.get('Priority', 1)
            if (q_type := q_data["queue_type"]) == 'Route':
                # TODO: support multi_dest
                dest = q_data.get("route_destinations").split(',')[0].split('@', maxsplit=1)
                if len(dest) == 1:
                    continue  # TODO: support local dest
                else:
                    queue_location = api.get_location(dest[1])
                    q_data = api.info(f'Queue/{dest[0]}', location=queue_location)['data']
            else:
                queue_location = api.location
            self.data[name] = Queue(
                Name=name,
                Type=q_type,
                Location=queue_location,
                Teams=q_data['teams'],
                Apps=q_data['apps'],
                Priority=q_pri,
                CoresPerHost=jmespath.search('resources_available.ncpu_perhost', q_data),
                CoresPerSocket=cores_per_socket,
                CoresPerNuma=cores_per_vnode,
                GpuPerNuma=jmespath.search('resources_available.ngpu_pernode', q_data) or 0,
                MemPerHost=jmespath.search('resources_available.mem_perhost', q_data),
                MinCores=jmespath.search('statistics.min_cores', q_data),
                MaxCores=jmespath.search('statistics.max_cores', q_data),
                MinGPUs=jmespath.search('statistics.min_gpus', q_data) or 0,
                MaxGPUs=jmespath.search('statistics.max_gpus', q_data) or 0,
                OfflineCores=jmespath.search('statistics.offline_cores', q_data),
                OfflineGPUs=jmespath.search('statistics.offline_gpus', q_data),
                RunningJobs=jmespath.search('statistics.running_jobs', q_data),
                UsingCores=jmespath.search('statistics.using_cores', q_data),
                UsingGPUs=jmespath.search('statistics.using_gpus', q_data) or 0,
                WaitingCores=jmespath.search('statistics.waiting_cores', q_data),
                Load=jmespath.search('statistics.load', q_data),
                FreeCoresGroup=jmespath.search('statistics.free_cores_group', q_data) or [],
                FreeGPUsGroup=jmespath.search('statistics.free_gpus_group', q_data) or []
            )

    def __hash__(self):
        return id(self)

    @cache
    def valid_queues(self,
                     app: Optional[str] = None,
                     use_gpu: Optional[bool] = None,
                     user_group: Optional[frozenset] = groups) -> List[Queue]:
        queues = []
        for q, d in self.data.items():
            teams = set(d.Teams)
            apps = set(d.Apps)
            apps.add(None)
            conditions = [
                'all' in teams or user_group & teams or is_admin,
                app in apps
            ]
            if use_gpu is True:
                conditions.append(d.has_gpu)
            elif use_gpu is False:
                conditions.append(not d.has_gpu)
            if all(conditions):
                queues.append(d)
        return queues

    @cache
    def recommend(self,
                  app: AppModel,
                  cpu: Optional[int] = None,
                  gpu: Optional[int] = None,
                  queue: Optional[str] = None,
                  job_count: int = 1,
                  user_group: Optional[frozenset] = groups) -> Tuple[int, int, Queue]:
        """
        recommend queue and cores

        app: app module
        cpu: user select cpu
        gpu: user select gpu
        queue: user select queue
        job_count: count of job
        user_group: user group

        ret:
            target_cores, gpu_opt, target_queue
        """
        can_use_gpu = app.DefaultGPU != 0
        can_use_cpu = app.DefaultMinCores != 0
        use_gpu = bool(gpu) or (can_use_gpu and not can_use_cpu) or None
        gpu_flag = 0
        queues = self.valid_queues(app.Name, use_gpu=use_gpu, user_group=user_group)
        logger.debug(f'valid queues are {[q.Name for q in queues]}')
        if not queues:
            raise SubError(f"can not find a queue for {app.Name}")

        def recommend_queue(use_cores):
            if queue:
                return self.data[queue]
            scores = Counter()
            for q in queues:
                if q.Priority <= -100:
                    continue
                scores.update(
                    {q: q.priority_score(use_cores, use_gpu=use_gpu, tiny_job=app.TinyJob, job_count=job_count)})
            rq = max(scores, key=scores.get)
            logger.debug(f'recommend queue is {rq}')
            return rq

        if queue_has_gpu := (self.data[queue].has_gpu if queue else None):
            if cpu:
                raise SubError(f'queue {queue} is only for gpu computing, please use -g to set gpu cores')
            elif not can_use_gpu:
                raise SubError(f'queue {queue} is only for gpu computing, {app.Name} can not run with gpu')
            cores = gpu or app.DefaultGPU
            target_queue = self.data[queue]
            gpu_flag = app.DefaultCoreWithGPU

        elif can_use_cpu and not gpu:
            default_cores = cpu or app.DefaultMinCores
            target_queue = recommend_queue(default_cores)
            cores = target_queue.recommend_cores(min_cores=default_cores, tiny_job=app.TinyJob)
            if cores != default_cores:
                logger.debug(f"changing cores from {default_cores} to {cores}")

        elif can_use_gpu:
            if queue_has_gpu is False:
                raise SubError(f"no gpu in {queue}")
            cores = gpu or app.DefaultGPU
            target_queue = recommend_queue(cores)
            gpu_flag = app.DefaultCoreWithGPU
        else:
            raise SubError(f'must choose cpu or gpu')
        return cores, gpu_flag, target_queue

    def show(self, show_config=False):
        """
        show queue statistics info
        """
        if show_config:
            data = [['Queue', 'Applications']]
        else:
            data = [['Queue', 'Jobs', 'FreeCores', 'WaitingCores', 'UsingCores', 'MinCores', 'MaxCores', 'OffCores']]
        for q, d in self.data.items():
            if not (d in self.valid_queues() or is_admin):
                continue
            if show_config:
                data.append([q, ','.join(d.Apps)])
            else:
                if d.has_gpu:
                    data.append([q, d.RunningJobs, f'{d.free_core_expr(use_gpu=True)}(G)', f'{d.WaitingGPUs}(G)',
                                 f'{d.UsingGPUs}(G)', f'{d.MinGPUs}(G)', f'{d.MaxGPUs}(G)', f'{d.OfflineGPUs}(G)'])
                else:
                    data.append(
                        [q, d.RunningJobs, d.free_core_expr(), d.WaitingCores, d.UsingCores, d.MinCores, d.MaxCores,
                         d.OfflineCores])
        print(AsciiTable(data).table)


class Job:
    """
    pbs job model
    """
    queue_info = Queues()

    def __init__(self, job_id: str, use_cache=True):
        if use_cache:
            data_raw = api.info(f'Jobs/{job_id}')
            logger.debug(f'get {job_id} data from pbs cache api')
            data = data_raw['data']
        else:
            logger.debug(f'get {job_id} data from pbs')
            data_raw = subprocess.getoutput(f'/opt/pbs/bin/qstat -F json -fx {job_id}')
            data = jmespath.search(f'Jobs.{job_id}', json.loads(data_raw))
        self.Id = job_id
        self.Name = data['Job_Name']
        self.State = JobState(data['job_state'])
        self.Placement = Placement.load(jmespath.search('Resource_List.place', data))
        self.Select = Select.load(jmespath.search('Resource_List.select', data))
        self.Queue = self.queue_info[data['queue']]
        self.Priority = data['Priority']
        self.Rerunable = data['Rerunable'] == 'True'

    def __str__(self):
        return self.Id


class Jobs:
    """
    PBS job data
    """

    @staticmethod
    def unfinished_jobs(*jobs) -> set:
        """
        return still running jobs in *jobs
        """
        current_jobs = subprocess.getoutput('/opt/pbs/bin/qselect -s QHRBS').split()
        return set(current_jobs) & set(*jobs)


class WallTime(BaseModel):
    hour: int = 0
    min: int = 0
    second: int = 0

    def __str__(self):
        return f'{self.hour:02}:{self.min:02}:{self.second:02}'


def placement_policy(queue: Queue,
                     cores: int,
                     gpu_opt: int = 0,
                     mpi: bool = False,
                     openmp_opt: int = 0,
                     memory_opt: int = 0) -> Tuple[Select, Placement, Dict[str, int]]:
    """
    set select and place argument

    queue: pbs queue
    cores: cpu or gpu cores
    gpu_opt:
        0: do not use gpu
        -1: use all cpu with gpu
        >=1: cpu number with one gpu
    mpi: mpi flag
    openmp:
        0: disable openmp
        -1: all cpus as omp threads
        -2: all cpus within a socket
        -3: all cpus within a numa node
        >=1: number of omp threads
    memory_opt:
        0: do not request memory
        -1: use all memory with cpu or gpu, unit: GB
        >0: total request memory, unit: GB
    """
    logger.debug(
        f'policy entry: cores: {cores}, gpu_opt: {gpu_opt}, mpi: {mpi}, openmp: {openmp_opt}, memory: {memory_opt}')
    variable = {}
    place = Placement()
    if gpu_opt:  # GPU job
        gpus = cores
        place.arrangement = Arrangement.VScatter
        select = Select(vnode_num := ceil(cores / (gpu_per_numa := queue.GpuPerNuma)))
        select.ngpus = gpu_per_numa if vnode_num > 1 else gpus
        select.ncpus = queue.CoresPerNuma * cores // gpu_per_numa if gpu_opt == -1 else gpu_opt * select.ngpus
        if gpus <= gpu_per_numa:
            place.group = ResourceGroup.Numa
        elif gpus <= queue.GpuPerSocket:
            place.group = ResourceGroup.Socket
        elif gpus <= queue.GpuPerHost:
            place.group = ResourceGroup.Host
        else:
            if not mpi:
                raise SubError("your job can not run across multiple servers")
            place.group = ResourceGroup.IBSwitch
        variable['SUB_GPU'] = cores
        variable['SUB_CPU'] = vnode_num * select.ncpus
        if memory_opt := memory_opt:
            numa_count = queue.numa_count
            nodes = min([vnode_num, numa_count])
            if memory_opt == -1:
                select.mem = queue.MemPerNuma
                variable['SUB_MEM'] = nodes * select.mem
            elif memory_opt == -2:
                variable['SUB_MEM'] = nodes * select.mem
            else:
                server_count = ceil(vnode_num / numa_count)
                variable['SUB_MEM'] = memory_opt // server_count
    else:  # CPU job
        cpus = cores
        select = Select(host_num := ceil(cpus / (cores_per_host := queue.CoresPerHost)))
        select.ncpus = cores_per_host if host_num > 1 else cpus
        if cpus <= queue.CoresPerNuma:
            place.group = ResourceGroup.Numa
        elif cpus <= queue.CoresPerSocket:
            place.group = ResourceGroup.Socket
        elif cpus <= queue.CoresPerHost:
            place.arrangement = Arrangement.Pack
        else:
            if not mpi:
                raise SubError("your job can not run across multiple servers")
            place.group = ResourceGroup.IBSwitch
        variable['SUB_CPU'] = select.select * select.ncpus
        if memory_opt := memory_opt:
            core_to_mem = int(select.ncpus * queue.MemPerCore)
            if memory_opt == -1:
                select.mem = core_to_mem
                variable['SUB_MEM'] = core_to_mem
            elif memory_opt == -2:
                variable['SUB_MEM'] = core_to_mem
            else:
                select.mem = memory_opt // select.select
                variable['SUB_MEM'] = select.mem
    if openmp_opt == -1:
        select.ompthreads = select.ncpus
    elif openmp_opt == -2:
        select.ompthreads = queue.CoresPerSocket
    elif openmp_opt == -3:
        select.ompthreads = queue.CoresPerNuma
    else:
        select.ompthreads = openmp_opt
    if select.ompthreads > select.ncpus:
        raise SubError(f"number of openmp thread should be less then all cpus")
    if mpi:
        select.mpiprocs = select.ncpus // select.ompthreads if select.ompthreads else select.ncpus
    return select, place, variable


class PBSCmdArgs:
    """
    pbs command line arguments
    """
    _para_map = {
        'jobname': 'N',
        'waitfor': 'W',
        'after': 'W',
        'umask': 'W',
        'group': 'W',
        'reservation': 'W',
        'priority': 'p',
        'select': 'l',
        'place': 'l',
        'walltime': 'l',
        'email': 'M',
        'project': 'P',
        'variable': 'v',
        'output': 'o',
        'at': 'a',
        'rerun': 'r',
        'remove': 'R',
        'checkpoint': 'c',
        'range': 'J'
    }

    _flag_map = {}

    def __init_subclass__(cls, **kwargs):
        if cls._para_map is not PBSCmdArgs._para_map:
            cls._para_map = {**PBSCmdArgs._para_map, **cls._para_map}

    def __init__(self):
        self._parameters = {}
        self._flags = set()

    def __setattr__(self, key, value):
        if isinstance(value, PbsRes):
            self._parameters[key] = 'l'
        elif key in self._para_map:
            self._parameters[key] = self._para_map[key]
        elif key in self._flag_map and value:
            self._flags.add(self._flag_map[key])
        super().__setattr__(key, value)

    def __delattr__(self, item):
        if item in self._parameters:
            self._parameters.pop(item, None)
        elif item in self._flags:
            self._flags.remove(item)
        super().__delattr__(item)

    def __str(self, key) -> str:
        value = getattr(self, key, None)
        if value is None:
            raise ValueError(f'invalid key {key}')
        if isinstance(value, PbsRes):
            return f'{key}={value.value}'
        if handler := getattr(self, f'_{key}', None):
            return handler(value)
        return str(value)

    def __repr__(self):
        return self._parameters

    @staticmethod
    def _waitfor(jids: List[str]) -> str:
        return f"depend=afterok:{':'.join(jids)}" if jids else ""

    @staticmethod
    def _after(jids: List[str]) -> str:
        return f"depend=after:{':'.join(jids)}" if jids else ""

    @staticmethod
    def _walltime(walltime: WallTime):
        return f'walltime={walltime}'

    @staticmethod
    def _group(group: str):
        return f'group_list={group}'

    @staticmethod
    def _project(p: str):
        return re.sub(r'\W', '', p).lower()

    @staticmethod
    def _reservation(r: bool):
        return f'create_resv_from_job={str(r).lower()}'

    @staticmethod
    def _rerun(value: bool):
        return 'y' if value else 'n'

    @staticmethod
    def _range(sequence: Union[range, List[int]]):
        logger.debug(f"pbs range is {sequence}")
        if isinstance(sequence, range):
            base = f'{sequence.start}-{sequence.stop - 1}'
            if (step := sequence.step) > 1:
                base += f':{step}'
        elif isinstance(sequence, list):
            if len(sequence) < 2:
                raise SubError("PBS range should be greater than 1")
            c_min, c_max = min(sequence), max(sequence)
            if c_min == c_max or set(sequence) != set(range(c_min, c_max + 1)):
                raise SubError("invalid PBS range")
            base = f'{c_min}-{c_max}'
        else:
            raise ValueError("invalid pbs range type")
        return base

    @staticmethod
    def _variable(value: Dict):
        return ','.join(f'{k}="{v}"' for k, v in value.items())

    @property
    def params(self):
        return ' '.join(f'-{flag} {value}' for key, flag in self._parameters.items() if (value := self.__str(key)))

    @property
    def flags(self):
        return ' '.join(f'-{f}' for f in self._flags)

    def dump(self):
        result = {}
        for i in self._parameters.keys():
            if attr := getattr(self, i, None):
                result[i] = attr
        return result


@Context
class Qsub(PBSCmdArgs):
    """
    qsub object wrapper
    """
    _para_map = {
        'queue': 'q',
    }

    _flag_map = {
        'hold': 'h'
    }

    def __init__(self):
        super().__init__()
        self.script = None
        self.remote = True
        self.workdir = os.getcwd()
        self.sudo = None
        self.waitfor = []
        self.after = []
        self.variable = {}
        self.cores = 0
        self.gpu = 0
        self.mpi = False
        self.openmp = 0
        self.memory = 0

    @staticmethod
    def _queue(q: Queue):
        return f'{q}@{local}' if local else str(q)

    @property
    def remote(self):
        return self._remote

    @remote.setter
    def remote(self, value: bool):
        self._remote = '--' if value else ''

    def policy(self):
        select, place, variable = placement_policy(self.queue, self.cores, self.gpu, self.mpi, self.openmp, self.memory)
        self.select = select
        self.place = place
        self.variable.update(variable)

    @property
    def command(self):
        cmd = f'/opt/pbs/bin/qsub {self.flags} {self.params} {self.remote} {self.script}'
        if self.sudo:
            cmd = f'sudo -u {self.sudo} {cmd}'
        return cmd

    def dump(self):
        result = super().dump()
        result['script'] = self.script
        result['workdir'] = self.workdir
        return result

    def run(self, test_info: Optional[list] = None, wait_jids=None) -> str:
        o_dir = os.getcwd()
        self.policy()
        if wait_jids:
            if isinstance(wait_jids, list):
                self.waitfor.extend(wait_jids)
            else:
                self.waitfor.append(wait_jids)
        logger.debug(cmd := self.command)
        if test_info is not None:
            logger.info("running in test mode")
            jid = f"{self.jobname}.{self.script.name}"
            test_info.append(self.dump())
        else:
            os.chdir(self.workdir)
            ret, jid = getstatusoutput(cmd)
            if ret != 0:
                os.chdir(o_dir)
                raise SubError("PBS qsub error")
        logger.info(f'jid for {self.jobname} is {Fore.GREEN}{jid}{Style.RESET_ALL}')
        return jid
