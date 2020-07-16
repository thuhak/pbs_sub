"""
main parser
"""
# author: thuhak.zhou@nio.com
from argparse import ArgumentParser
from weakref import WeakValueDictionary
import logging
from logging.handlers import RotatingFileHandler
import json
import os
from os.path import dirname, join, abspath, isfile, isdir
from string import ascii_letters, digits
import pwd
import time

import sh

__version__ = '1.0.0'
__author__ = 'thuhak.zhou@nio.com'


class classproperty:
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
    # fix pbs server config here
    PBS_SERVER = 'PBS_SERVER'
    MAIL = 'xx.com'
    QSUB = sh.Command('/opt/pbs/bin/qsub')
    user = pwd.getpwuid(os.getuid())[0]
    script_base = join(dirname(abspath(__file__)), 'run_scripts')

    def __init__(self):
        self.base_args = []
        self.parser = ArgumentParser(description=f"This script is used for pbs job submission, version: {__version__}",
                                     epilog=f'if you have any question or suggestion, please contact with {__author__}')
        self.parser.add_argument('-n', '--name', help='job name')
        self.parser.add_argument('-l', '--log_level', choices=['error', 'info', 'debug'], default='info',
                                 help='logging level')
        self.parser.add_argument('-W', '--wait', metavar='JOB_ID', help='depend on which job')
        self.parser.add_argument('--free_cores', action='store_true', help='show free cpu cores by queue')
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
    def handle(cls, args, base_args) -> list:
        """
        abc interface, implement this method in subclass

        args: argument args
        base_args: all argument provided by main parser
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
            PBSNODES = sh.Command('/opt/pbs/bin/pbsnodes')
            raw_nodedata = json.loads(PBSNODES('-aS', '-F', 'json').stdout)['nodes']
            raw_nodedata2 = json.loads(PBSNODES('-Saj', '-F', 'json').stdout)['nodes']
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
            QSTAT = sh.Command('/opt/pbs/bin/qstat')
            raw_data = sh.grep(QSTAT('-f', '-F', 'json'), '-v', 'Submit_arguments').stdout
            job_data = json.loads(raw_data)['Jobs']
            return job_data
        except Exception as e:
            cls.logger.error(f'PBS error, can not get pbs job info, reason: {str(e)}')
            exit(-1)

    @classmethod
    def free_cores(cls, queue: str) -> int:
        """
        get free cores in queue
        """
        count = 0
        cls.logger.debug(f'getting free cores in {queue}')
        pbsdata = cls.pbs_nodes_data
        for node in pbsdata.values():
            if node.get('queue') == queue:
                core_free = int(node.get('ncpus f/t').split('/')[0])
                count += core_free
        cls.logger.debug(f'number of free cores in {queue} is {count}')
        return count

    @classmethod
    def all_free_cores(cls) -> dict:
        """
        get all free cores in all queues
        """
        from collections import defaultdict
        cls.logger.debug('getting free cores')
        pbsdata = cls.pbs_nodes_data
        ret = defaultdict(int)
        for node in pbsdata.values():
            queue = node.get('queue')
            if queue:
                ret[queue] += int(node.get('ncpus f/t').split('/')[0])
        return ret

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
            QSTAT = sh.Command('/opt/pbs/bin/qstat')
            raw_info = sh.grep(QSTAT('-f', '-F', 'json', jid), '-v', 'Submit_arguments').stdout #pbs bug fix
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

    @staticmethod
    def get_ncpu(queue: str) -> int:
        return 16 if queue == 'cfdbs' else 32

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
            loghandler = RotatingFileHandler(join(logdir, self.user + '.log'), maxBytes=10*1024*1024, backupCount=3, encoding='utf-8')
            handlers.append(loghandler)
        logging.basicConfig(level=log_level, format='%(asctime)s [%(levelname)s]: %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S", handlers=handlers)
        for handler in logging.root.handlers:
            handler.addFilter(logging.Filter('pbs_sub'))
        if args.free_cores:
            free_cores = self.all_free_cores()
            ali = max([len(x) for x in free_cores.keys()]) + 3
            for q, c in free_cores.items():
                space = ' ' * (ali - len(q))
                print(f'{q}:{space}{c}')
            return
        software = args.software
        email = f'{self.user}@{self.MAIL}'
        waitfor = args.wait
        base_args = ['-m', 'abe', '-M', email]
        if waitfor:
            if '.' not in waitfor:
                waitfor = waitfor + '.' + self.PBS_SERVER
            if not self.check_jobid(waitfor):
                self.logger.debug(f'invalid job id {waitfor}')
                exit(1)
            var_w = f'depend=afterok:{waitfor}'
            base_args.extend(['-W', var_w])
        if software in self._all_software:
            jids = self._all_software[software].handle(args, base_args)
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
        else:
            self.parser.print_help()
