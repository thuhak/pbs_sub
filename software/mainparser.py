"""
main parser
"""
# author: thuhak.zhou@nio.com
import json
import logging
import os
import pwd
import subprocess
import time
from argparse import ArgumentParser
from collections import namedtuple, OrderedDict
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from math import ceil
from os.path import dirname, join, abspath, isfile, isdir
from string import ascii_letters, digits
from weakref import WeakValueDictionary

import jmespath

from .database import session_scope, Job

__version__ = '4.3.0'
__author__ = 'thuhak.zhou@nio.com'

user = pwd.getpwuid(os.getuid())[0]
groups = set(subprocess.getoutput('groups').split())

pbs_servers = {
    'A': 'sh',
    'B': 'hf'
}

with open('/etc/pbs.conf') as f:
    for l in f:
        if l.startswith('PBS_SERVER'):
            local_server = l.split('=')[1].strip()
            break

server = pbs_servers[local_server]
Param = namedtuple('Param', ['flag', 'name', 'value'])


def jid_to_int(jid: str) -> int:
    return int(jid.split('.')[0])


class Qsub:
    """
    qsub command
    """
    para_map = {
        'jobname': 'N',
        'waitfor': 'W',
        'umask': 'W',
        'group': 'W',
        'priority': 'p',
        'select': 'l',
        'place': 'l',
        'resource': 'l',
        'queue': 'q',
        'email': 'M',
        'project': 'P',
        'variable': 'v',
        'output': 'o',
        'date': 'a',
        'rerun': 'r',
        'remove': 'R',
        'checkpoint': 'c'
    }

    flag_map = {
        'hold': 'h'
    }

    def __init__(self, logger, script=None, remote=True, server=None):
        self.parameters = OrderedDict()
        self.flags = set()
        self.script = script
        self.logger = logger
        self.remote = remote
        self.server = server
        self.software = None
        self.jobfile = None
        self.cores = None

    def __setattr__(self, key, value):
        if key in self.para_map:
            handler = getattr(self, f'_{key}', None)
            if handler:
                value = handler(value)
            parm = Param(self.para_map[key], key, value)
            super().__setattr__(key, value)
            self.parameters[key] = parm
        elif key in self.flag_map and value:
            self.flags.add(self.flag_map[key])
        else:
            super().__setattr__(key, value)

    def __delattr__(self, item):
        if item in self.parameters:
            self.parameters.pop(item, None)
        elif item in self.flags:
            self.flags.remove(item)
        super().__delattr__(item)

    def _waitfor(self, jid):
        return f'depend=afterok:{jid}'

    def _group(self, group):
        return f'group_list={group}'

    def _queue(self, q):
        return f'{q}@{self.server}' if self.server else q

    @property
    def remote(self):
        return self._remote

    @remote.setter
    def remote(self, value):
        self._remote = '--' if value else ''

    @property
    def command(self):
        flags = ' '.join(f'-{f}' for f in self.flags)
        params = ' '.join(f'-{p.flag} {p.value}' for p in self.parameters.values())
        cmd = f'/opt/pbs/bin/qsub {flags} {params} {self.remote} {self.script}'
        self.logger.debug(cmd)
        return cmd

    def run(self, **kwargs):
        ret, data = subprocess.getstatusoutput(self.command)
        if ret != 0:
            self.logger.error(f'qsub error: {data}')
            exit(-1)
        jid = jid_to_int(data)
        self.logger.info(f'your job id is {jid}')
        self.logger.debug('updating database')
        db_extra_args = {k: v for k, v in kwargs.items() if k in Job.__mapper__.c}
        for k in ('software', 'project', 'queue', 'jobfile', 'cores'):
            if k not in db_extra_args:
                db_extra_args[k] = getattr(self, k, None)
        try:
            with session_scope() as sess:
                record = Job(jid=jid, user=user, **db_extra_args)
                sess.add(record)
        except Exception as e:
            self.logger.error(f'database error, error: {str(e)}')
        return data


class classproperty:
    """
    cache data in class
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if cls is None:
            return self
        else:
            value = self.func(cls)
            setattr(cls, self.func.__name__, value)
            return value


@lru_cache()
def pbs_nodes_data(server=None):
    server_flag = f"-s {server}" if server else ""
    raw_nodedata = json.loads(subprocess.getoutput(f'/opt/pbs/bin/pbsnodes {server_flag} -av -F json'))['nodes']
    raw_nodedata2 = json.loads(subprocess.getoutput(f'/opt/pbs/bin/pbsnodes {server_flag} -Sajv -F json'))['nodes']
    for k in raw_nodedata.keys():
        raw_nodedata[k].update(raw_nodedata2[k])
    return raw_nodedata


@lru_cache()
def pbs_job_data(server=None):
    server_flag = f'@{server}' if server else ""
    return json.loads(subprocess.getoutput(f'/opt/pbs/bin/qstat {server_flag} -f -F json'))['Jobs']


@lru_cache()
def pbs_queue_data(server=None):
    server_flag = f'@{server}' if server else ""
    return json.loads(subprocess.getoutput(f'/opt/pbs/bin/qstat {server_flag} -Qf -F json'))["Queue"]


@lru_cache()
def pbs_server_data(server=None):
    server_flag = server if server else ""
    return [*json.loads(subprocess.getoutput(f'/opt/pbs/bin/qstat {server_flag} -Bf -F json'))['Server'].values()][0]


class MainParser:
    """
    provide common args for pbs.
    argument parser of subclasses will register in _all_software variable
    the request will be routed to the right software subclass
    """
    _all_software = WeakValueDictionary()
    logger = logging.getLogger('pbs_sub')
    script_base = join(dirname(abspath(__file__)), 'run_scripts')
    local_server = server

    def __init__(self):
        self.base_args = []
        self.parser = ArgumentParser(description=f"This script is used for pbs job submission, version: {__version__}",
                                     epilog=f'if you have any question or suggestion, please contact with {__author__}')
        self.parser.add_argument('-n', '--name', help='job name')
        self.parser.add_argument('-l', '--log_level', choices=['error', 'info', 'debug'], default='info',
                                 help='logging level')
        self.parser.add_argument('-W', '--wait', metavar='JOB_ID', help='depend on which job')
        self.parser.add_argument('-s', '--server', choices=['sh', 'hf'], help='which server you wanna connect')
        self.parser.add_argument('-p', '--priority', type=int, help='priority of job, range:-1024-1023')
        self.parser.add_argument('-P', '--project', help='name of your project, default is the name of the software')
        self.parser.add_argument('--keep', action='store_true', help='keep running until job done')
        self.parser.add_argument('--hold', action='store_true', help='hold job after submission')
        software = self.parser.add_subparsers(dest='software', help='software list')
        for soft in self._all_software:
            cls = self._all_software[soft]
            parser = software.add_parser(cls.__software__, help=f'(script version {cls.__version__})')
            cls.add_parser(parser)

    @classmethod
    def add_parser(cls, parser: ArgumentParser):
        """
        implement this method in subclass if argument is not as same as the fault
        """
        parser.add_argument('-c', '--core', type=int, help='how many cores you need')
        parser.add_argument('-q', '--queue', help='queue of job')
        parser.add_argument('jobfile', help='jobfile')

    @classmethod
    def handle(cls, args, qsub: Qsub) -> list:
        """
        abc interface, implement this method in subclass

        args: argument args
        base_args: qsub instance
        :return all job ids in list format, or you may just return None
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        """
        register subclass
        """
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__software__') or not getattr(cls, '__version__'):
            raise NotImplementedError('you need to set __software__ and __version__ attribute in your subclass')
        cls._all_software[cls.__software__] = cls

    @classproperty
    def default_run(cls):
        """
        default run script
        """
        run_script = join(cls.script_base, cls.__software__ + '.sh')
        if not isfile(run_script):
            cls.logger.error(f'you need to put the script {run_script} first')
            exit(1)
        return run_script

    @classmethod
    def set_project(cls, qsub: Qsub, jobfile: str):
        """
        guess project name by the path of jobfile

        pre config in pbs
        qmgr -c "create resource Project type = string_array, flag=h"
        qmgr -c "set server resources_available.Project = 'proj1, proj2,...'"
        """
        if getattr(qsub, 'project', None):
            return
        server = qsub.server
        _jobfile = jobfile.lower()
        try:
            projects = jmespath.search('resources_available.Project', pbs_server_data(server)).split(",")
            cls.logger.debug(f"builtin projects are {projects}")
            for p in projects:
                if p.lower() in _jobfile:
                    qsub.project = p
                    return
            qsub.project = 'unknown'
        except Exception as e:
            cls.logger.error(f"set project error, reason {str(e)}")

    @classmethod
    def all_free_cores(cls, server=None):
        """
        get all free cores in all queues
        """
        from collections import defaultdict
        cls.logger.debug('getting free cores')
        pbsdata = pbs_nodes_data(server)
        jobdata = pbs_job_data(server)
        queue_info = defaultdict(lambda: defaultdict(int))
        for k, node in pbsdata.items():
            queues_raw = node['resources_available'].get('Qlist')
            queue_raw = node.get('queue')
            if queues_raw:
                queues = queues_raw.split(',')
            elif queue_raw:
                queues = [queue_raw]
            else:
                continue
            cpu_free, cpu_total = [int(x) for x in node.get('ncpus f/t').split('/')]
            qlen = len(queues)
            for q in queues:
                if node.get('State') in ('down', 'offline'):
                    queue_info[q]['offline'] += cpu_total
                    continue
                queue_info[q]['free'] += cpu_free
                queue_info[q]['total'] += cpu_total
                if qlen == 1:
                    queue_info[q]['private'] += cpu_total
        for job, data in jobdata.items():
            queue = data['queue']
            status = data['job_state']
            cpus = data["Resource_List"]["ncpus"]
            if status == 'R':
                queue_info[queue]['running_jobs'] += 1
                queue_info[queue]['using_cpus'] += cpus
            elif status == 'Q':
                queue_info[queue]['waiting_jobs'] += 1
                queue_info[queue]['waiting_cpus'] += cpus
        return queue_info

    @classmethod
    def check_jobid(cls, jid: str, server=None) -> bool:
        """
        check job id in pbs or not
        """
        cls.logger.debug(f'checking job {jid}')
        return jid in pbs_job_data(server)

    @classmethod
    def get_jid_info(cls, jid: str) -> dict:
        cls.logger.debug(f'getting job information for {jid}')
        try:
            raw_info = subprocess.getoutput(f'/opt/pbs/bin/qstat -fx -F json {jid}')
            job_info = json.loads(raw_info)['Jobs'][jid]
            return job_info
        except json.decoder.JSONDecodeError as e:
            cls.logger.error(f'job {jid} is not json format')
            cls.logger.debug(str(e))
            exit(1)
        except Exception as e:
            cls.logger.error(f'unable to get information from pbs, reason: {str(e)}')
            exit(1)

    @staticmethod
    def fix_jobname(name: str) -> str:
        valid_str = ascii_letters + digits + '_-'
        ret = n = name.rsplit('.', maxsplit=1)[0]
        for c in n:
            if c not in valid_str:
                ret = ret.replace(c, '_')
        return ret

    @classmethod
    def get_queue_res(cls, queue: str, res: str, default=0, server=None, item='resources_available') -> int:
        """
        pre config in pbs
        qmgr -c "create resource RES type=long, flag=h"
        qmgr -c "set queue QUEUE resources_available.RES = NUM"
        """
        ret = jmespath.search(f'"{queue}".{item}."{res}"', pbs_queue_data(server))
        if not ret:
            ret = default
        return ret

    @classmethod
    def set_select(cls, cores: int, queue: str, qsub: Qsub, mpi=False, set_mem=False, gpu=False, memory=None, omp=None):
        """
        pre config in pbs

        qmgr -c "create resource ncpu_perhost type=long, flag=h"
        qmgr -c "create resource mem_perhost type=long, flag=h"
        qmgr -c "create resource ncpu_pernuma type=long, flag=h"
        qmgr -c "create resource ncpu_pernode type=long, flag=h"
        qmgr -c "create resource ngpu_pernode type=long, flag=h"
        qmgr -c "create resource ngpu_pernode type=long, flag=h"

        and set them in each queue

        qmgr -c "create resource numa type=string, flag=h"
        qmgr -c "create resource ibswitch type=string, flag=h"

        and set them in each vnode, add numa and ibswitch resources in sched_config, then reload scheduler
        """
        server = qsub.server
        cores_per_host = cls.get_queue_res(queue, 'ncpu_perhost', server=server)
        cores_per_numa = cls.get_queue_res(queue, 'ncpu_pernuma', server=server)
        cores_per_vnode = cls.get_queue_res(queue, 'ncpu_pernode', server=server)
        mem_per_host = cls.get_queue_res(queue, 'mem_perhost', server=server)
        try:
            host_num = ceil(cores / cores_per_host)
        except ZeroDivisionError:
            cls.logger.error(f'queue config error, contact with {__author__}')
            exit(1)
        if host_num > 1:
            if not mpi:
                cls.logger.error("your job does not support run on multiple servers")
                exit(1)
            place = 'place=group=ibswitch'
            core = cores_per_host
            mem = memory / host_num if memory else mem_per_host
        else:
            place = 'place=pack'
            core = cores
            mem = int(mem_per_host * core / cores_per_host) - 1
        if cores_per_vnode < cores <= cores_per_numa:
            place += ':group=numa'
        elif cores <= cores_per_vnode:
            place += ':group=vnode'
        if gpu:
            gpu_per_node = cls.get_queue_res(queue, 'ngpu_pernode', server=server)
            gpus = core
            core = core * cores_per_vnode // gpu_per_node
            select = f'select={host_num}:ngpus={gpus}:ncpus={core}'
        else:
            select = f'select={host_num}:ncpus={core}'
        if mpi:
            select += f':mpiprocs={core}'
        if set_mem:
            select += f':mem={mem}gb'
        if omp:
            select += f':ompthreads={omp}'
        qsub.place = place
        qsub.select = select
        return core

    @classmethod
    def get_check_cores(cls, cores, max_cores, queue, server=None, mpi=True) -> int:
        """
        get default cores of queue if cores is None.
        otherwise, check cores number is available.
        """
        if cores:
            cores_per_host = cls.get_queue_res(queue, 'ncpu_perhost', server=server)
            cores_per_numa = cls.get_queue_res(queue, 'ncpu_pernuma', server=server)
            cores_per_vnode = cls.get_queue_res(queue, 'ncpu_pernode', server=server)
            error_conditions = [
                mpi is False and cores > cores_per_host,
                cores > cores_per_host and (cores > max_cores or cores % cores_per_host),
                cores_per_numa < cores < cores_per_host and cores % cores_per_numa,
                cores < cores_per_vnode
            ]
            if any(error_conditions):
                cls.logger.error('not valid cores')
                exit(1)
            return cores
        else:
            default_cpus = cls.get_queue_res(queue, 'ncpus', server=server, default=0, item='default_chunk')
            cls.logger.debug(f'default chunk of {queue} is {default_cpus}')
        return default_cpus or cls.get_queue_res(queue, 'ncpu_pernode', server=server)

    @classmethod
    def available_queues(cls, software=None, server=None) -> list:
        """
        pre config
        qmgr -c "create resource Team type=string_array, flag=h"
        qmgr -c "create resource App type=string_array, flag=h"
        """
        result = []
        queue_data = pbs_queue_data(server)
        for queue, data in queue_data.items():
            if software:
                apps_in_queue = jmespath.search(f'"{queue}".resources_available.App', queue_data)
                apps = set(apps_in_queue.split(',')) if apps_in_queue else set()
                if software not in apps:
                    continue
            teams_in_queue = jmespath.search(f'"{queue}".resources_available.Team', queue_data)
            teams = set(teams_in_queue.split(',')) if teams_in_queue else set()
            if groups & teams or 'all' in teams:
                result.append(queue)
        return result

    @classmethod
    def get_queue(cls, queue=None, software=None, server=None):
        """
        if queue is None, return the queue with most free cores
        """
        if not software:
            software = cls.__software__
        queues = cls.available_queues(software, server)
        if queue:
            if queue not in queues:
                cls.logger.error(f'invalid queue {queue}, available queues are {queues}')
                exit(1)
            else:
                return queue
        elif not queues:
            cls.logger.error(f'not available queue for your team, please contact with {__author__}')
            exit(1)
        else:
            queue_info = cls.all_free_cores(server)
            sorted_queue = sorted(queues, key=lambda q: queue_info[q]['free'], reverse=True)
            cls.logger.debug(f'choosing queue in {sorted_queue} which have most cores')
            q = sorted_queue[0]
            return q

    @classmethod
    def default_policy(cls, queue, cores, qsub, maxcores, jobfile=None, mpi=False, set_mem=False,
                       gpu=False, memory=None, omp=None, script=None) -> tuple:
        """
        default pbs job settings for most case
        """
        target_queue = cls.get_queue(queue, server=qsub.server)
        qsub.queue = target_queue
        target_cores = cores if gpu else cls.get_check_cores(cores, maxcores, target_queue, qsub.server)
        cpus = cls.set_select(target_cores, target_queue, qsub, mpi=mpi, set_mem=set_mem, gpu=gpu, memory=memory, omp=omp)
        qsub.script = script or cls.default_run
        qsub.cores = target_cores
        if jobfile:
            jobfile = os.path.abspath(jobfile)
            qsub.jobfile = jobfile
            if not os.path.exists(jobfile):
                cls.logger.error(f'{jobfile} does not exist')
                exit(1)
            workdir = jobfile if os.path.isdir(jobfile) else os.path.dirname(jobfile)
            os.chdir(workdir)
            qsub.output = workdir
            cls.set_project(qsub, jobfile)
            if not getattr(qsub, 'jobname', None):
                qsub.jobname = cls.fix_jobname(os.path.basename(jobfile))
        return target_queue, cpus, jobfile

    @classmethod
    def get_mem(cls, cores: int, queue: str, server=None) -> int:
        """
        get total memory by cores and queue
        """
        mem_per_host = cls.get_queue_res(queue, 'mem_perhost', server=server)
        cores_per_host = cls.get_queue_res(queue, 'ncpu_perhost', server=server)
        return int(mem_per_host * cores / cores_per_host)

    def run(self):
        """
        program entry
        """
        args = self.parser.parse_args()
        log_level = getattr(logging, args.log_level.upper())
        handlers = [logging.StreamHandler()]
        logdir = '/var/log/pbs_sub'
        if isdir(logdir):
            loghandler = RotatingFileHandler(join(logdir, user + '.log'), maxBytes=10 * 1024 * 1024, backupCount=3,
                                             encoding='utf-8')
            handlers.append(loghandler)
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter('pbs_sub'))
        server = args.server
        qsub = Qsub(self.logger, server=server)
        software = args.software
        qsub.email = user + '@nio.com'
        if args.hold:
            qsub.hold = True
        waitfor = args.wait
        if waitfor:
            qsub.waitfor = waitfor
        priority = args.priority
        if priority:
            if priority >= 1024 or priority < -1024:
                self.logger.error('priority out of range')
                exit(3)
            qsub.priority = priority
        if args.name:
            qsub.jobname = args.name
        if args.project:
            qsub.project = args.project
        if software in self._all_software:
            if user == 'root':
                self.logger.error('you can not submit job via root user, please use your own user')
                exit(127)
            qsub.software = software
            jids = self._all_software[software].handle(args, qsub)
            if not jids:
                return
            time.sleep(1)
            for jid in jids:
                job_info = self.get_jid_info(jid)
                job_stat = job_info.get('job_state')
                info = f'job {jid} is in state {job_stat}'
                comment = job_info.get('comment')
                if comment:
                    info += f',comment: {comment}'
                self.logger.info(info)
            while jids and args.keep:
                time.sleep(5)
                for jid in jids[:]:
                    job_info = self.get_jid_info(jid)
                    job_stat = job_info.get('job_state')
                    if job_stat in ('H', 'R', 'Q', 'S', 'E'):
                        self.logger.info(f'{jid} is not finished')
                        break
                    else:
                        jids.remove(jid)
        else:
            # show usage information
            from terminaltables import AsciiTable
            queue_info = self.all_free_cores(server)
            queues = self.available_queues(server)
            table_data = [
                ['queue', 'running_jobs', 'free_cores', 'waiting_cores', 'using_cores', 'private_cores', 'total_cores',
                 'offline_cores',
                 'load']]
            for q, data in queue_info.items():
                if q not in queues and user != 'root':
                    continue
                if data['total'] == 0:
                    load = 0
                else:
                    load = round((data['using_cpus'] + data['waiting_cpus']) * 100.0 / data['total'], 1)
                table_data.append(
                    [q, data['running_jobs'], data['free'], data['waiting_cpus'], data['using_cpus'], data['private'],
                     data['total'], data['offline'], f'{load}%'])
            table = AsciiTable(table_data)
            print(table.table)
