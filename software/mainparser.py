"""
main parser
"""
# author: thuhak.zhou@nio.com
import json
import os
import logging
import pwd
import time
import subprocess
import re
from argparse import ArgumentParser
from weakref import WeakValueDictionary
from logging.handlers import RotatingFileHandler
from os.path import dirname, join, abspath, isfile, isdir
from string import ascii_letters, digits
from collections import namedtuple, OrderedDict

import jmespath

__version__ = '4.1.0'
__author__ = 'thuhak.zhou@nio.com'

Param = namedtuple('Param', ['flag', 'name', 'value'])


class Qsub:
    """
    qsub command
    """
    para_map = {
        'jobname': 'N',
        'waitfor': 'W',
        'umask': 'W',
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
        'remove': 'R'
    }

    def __init__(self, logger, script=None, remote=False):
        self.parameters = OrderedDict()
        self.script = script
        self.logger = logger
        self.remote = '--' if remote else ''

    def __setattr__(self, key, value):
        if key in self.para_map:
            parm = Param(self.para_map[key], key, value)
            super().__setattr__(key, parm)
            self.parameters[key] = parm
        else:
            super().__setattr__(key, value)

    def __delattr__(self, item):
        self.parameters.pop(item, None)
        super().__delattr__(item)

    @property
    def command(self):
        params = ' '.join(f'-{p.flag} {p.value}' for p in self.parameters.values())
        return f'/opt/pbs/bin/qsub {params} {self.remote} {self.script}'

    def run(self):
        ret, data = subprocess.getstatusoutput(self.command)
        if ret != 0:
            self.logger.error(f'qsub error: {data}')
            exit(-1)
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


class MainParser:
    """
    provide common args for pbs.
    argument parser of subclasses will regist in _all_software variable
    the request will be routed to the right software subclass
    """
    _all_software = WeakValueDictionary()
    logger = logging.getLogger('pbs_sub')
    user = pwd.getpwuid(os.getuid())[0]
    script_base = join(dirname(abspath(__file__)), 'run_scripts')

    def __init__(self):
        self.base_args = []
        self.parser = ArgumentParser(description=f"This script is used for pbs job submission, version: {__version__}",
                                     epilog=f'if you have any question or suggestion, please contact with {__author__}')
        self.parser.add_argument('-n', '--name', help='job name')
        self.parser.add_argument('-N', '--node', help='job run at specific node')
        self.parser.add_argument('-l', '--log_level', choices=['error', 'info', 'debug'], default='info',
                                 help='logging level')
        self.parser.add_argument('-W', '--wait', metavar='JOB_ID', help='depend on which job')
        self.parser.add_argument('-p', '--priority', type=int, help='priority of job, range:-1024-1023')
        self.parser.add_argument('-P', '--project', help='name of your project, default is the name of the software')
        self.parser.add_argument('--free_cores', action='store_true', help='show free cpu cores by queue')
        self.parser.add_argument('--keep', action='store_true', help='keep running until job done')
        software = self.parser.add_subparsers(dest='software', help='software list')
        for soft in self._all_software:
            cls = self._all_software[soft]
            parser = software.add_parser(cls.__software__, help=f'(script version {cls.__version__})')
            cls.add_parser(parser)

    @classmethod
    def add_parser(cls, parser: ArgumentParser):
        """
        abc interface, implement this method in subclass
        """
        raise NotImplementedError

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
        regist subclass
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

    @classproperty
    def pbs_nodes_data(cls):
        """
        get node data from pbs
        """
        cls.logger.debug('getting pbs node info')
        try:
            raw_nodedata = json.loads(subprocess.getoutput('/opt/pbs/bin/pbsnodes -av -F json'))['nodes']
            raw_nodedata2 = json.loads(subprocess.getoutput('/opt/pbs/bin/pbsnodes -Sajv -F json'))['nodes']
            for k in raw_nodedata.keys():
                raw_nodedata[k].update(raw_nodedata2[k])
            return raw_nodedata
        except Exception as e:
            cls.logger.error(f'PBS error, can not get pbs node info, reason: {str(e)}')
            exit(-1)

    @classproperty
    def pbs_job_data(cls):
        """
        get pbs job data
        """
        cls.logger.debug('getting pbs job info')
        try:
            return json.loads(subprocess.getoutput(f'/opt/pbs/bin/qstat -f -F json'))['Jobs']
        except Exception as e:
            cls.logger.error(f'PBS error, can not get pbs job info, reason: {str(e)}')
            exit(-1)

    @classproperty
    def pbs_queue_data(cls):
        """
        get pbs queue setting info
        """
        cls.logger.debug('get pbs queue info')
        try:
            return json.loads(subprocess.getoutput('/opt/pbs/bin/qstat -Qf -F json'))["Queue"]
        except Exception as e:
            cls.logger.error(f'PBS error, can not get pbs queue info, reason: {str(e)}')
            exit(-1)

    @classproperty
    def PBS_SERVER(cls):
        """
        get current pbs server
        """
        with open('/etc/pbs.conf') as f:
            for l in f:
                if l.startswith('PBS_SERVER'):
                    return l.strip().split('=')[1]

    @classmethod
    def all_free_cores(cls) -> tuple:
        """
        get all free cores in all queues
        """
        from collections import defaultdict
        cls.logger.debug('getting free cores')
        pbsdata = cls.pbs_nodes_data
        free = defaultdict(int)
        total = defaultdict(int)
        offline = defaultdict(int)
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
            for q in queues:
                if node.get('State') in ('down', 'offline'):
                    offline[q] += cpu_total
                    continue
                free[q] += cpu_free
                total[q] += cpu_total
        return free, total, offline

    @classmethod
    def check_jobid(cls, jid: str) -> bool:
        """
        check job id in pbs or not
        """
        cls.logger.debug(f'checking job {jid}')
        return jid in cls.pbs_job_data

    @classmethod
    def get_jid_info(cls, jid: str) -> dict:
        cls.logger.debug(f'getting job information for {jid}')
        try:
            raw_info = subprocess.getoutput(f'/opt/pbs/bin/qstat -fx -F json {jid}')
            job_info = json.loads(raw_info)['Jobs'][jid]
            return job_info
        except json.decoder.JSONDecodeError:
            cls.logger.error(f'job {jid} is not json format')
            cls.logger.debug(raw_info)
            exit(1)
        except Exception as e:
            cls.logger.error(f'unable to get information from pbs, reason: {str(e)}')
            exit(1)

    @staticmethod
    def replace_id(args: list, jobid: str) -> None:
        """this function will change the state of args"""
        try:
            w_i = args.index('-W')
            args[w_i + 1] = f'depend=afterok:{jobid}'
        except ValueError:
            args.extend(['-W', f'depend=afterok:{jobid}'])

    @staticmethod
    def fix_jobname(name: str) -> str:
        valid_str = ascii_letters + digits + '_'
        ret = name
        for c in name:
            if c not in valid_str:
                ret = ret.replace(c, '_')
        return ret

    @classmethod
    def get_ncpu(cls, queue: str) -> int:
        """
        set default_chunk.cpus first
        """
        return jmespath.search(f'{queue}.default_chunk.ncpus', cls.pbs_queue_data)

    @classmethod
    def get_mem(cls, queue: str) -> int:
        """
        set default_chunk.mem = INTgb first
        """
        chunk = jmespath.search(f'{queue}.default_chunk.mem', cls.pbs_queue_data)
        return int(re.findall(r'\d+', chunk)[0])

    @staticmethod
    def jid_to_int(jid: str) -> int:
        return int(jid.split('.')[0])

    def run(self):
        """
        program entry
        """
        if self.user == 'root':
            self.logger.error('you can not submit job via root user, please use your own user')
            exit(127)
        args = self.parser.parse_args()
        log_level = getattr(logging, args.log_level.upper())
        handlers = [logging.StreamHandler()]
        logdir = '/var/log/pbs_sub'
        if isdir(logdir):
            loghandler = RotatingFileHandler(join(logdir, self.user + '.log'), maxBytes=10 * 1024 * 1024, backupCount=3,
                                             encoding='utf-8')
            handlers.append(loghandler)
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter('pbs_sub'))
        if args.free_cores:
            from terminaltables import AsciiTable
            free_cores, total_cores, offline_cores = self.all_free_cores()
            table_data = [['queue', 'free_cores', 'offline_cores', 'total_cores']]
            for q in total_cores.keys():
                table_data.append([q, free_cores[q], offline_cores[q], total_cores[q]])
            table = AsciiTable(table_data)
            print(table.table)
            return
        qsub = Qsub(self.logger)
        software = args.software
        qsub.email = self.user + '@nio.com'
        qsub.project = args.project or software
        waitfor = args.wait
        if waitfor:
            if '.' not in waitfor:
                waitfor = waitfor + '.' + self.PBS_SERVER
            if not self.check_jobid(waitfor):
                self.logger.debug(f'invalid job id {waitfor}')
                exit(1)
            qsub.waitfor = f'depend=afterok:{waitfor}'
        priority = args.priority
        if priority:
            if priority >= 1024 or priority < -1024:
                self.logger.error('priority out of range')
                exit(3)
            qsub.priority = priority
        node = args.node
        if node:
            qsub.resource = f'host={node}'
        if software in self._all_software:
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
            self.parser.print_help()
