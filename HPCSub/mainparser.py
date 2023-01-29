"""
sub command line main parser
"""
# author: thuhak.zhou@nio.com
import os
import re
import json
import time
import logging
from enum import Enum
from typing import List, Optional, Dict
from argparse import ArgumentParser, Action, Namespace

import chardet
import pyparsing as pp
from colorama import Fore, Style

from .config import *
from .pbs import *

__version__ = '6.0.1'
__author__ = 'thuhak.zhou@nio.com'


class FileCheckType(Enum):
    Exist = "must_exist"
    Remind = "remind"
    Executable = "executable"
    IsDir = "isdir"
    MakeDir = "makedir"


def check_file(action: FileCheckType, value: str, formats: Optional[List[str]] = None) -> HPCFile:
    hpc_file = HPCFile(value)
    file_str = str(hpc_file)
    if formats:
        for f in formats:
            if file_str.lower().endswith(f.lower()):
                break
        else:
            raise SubError(f'{hpc_file} should be endswith {formats}')
    if chardet.detect(file_str.encode())['encoding'] not in ('ascii', 'utf-8'):
        raise SubError(f'{file_str} contains illegal characters')
    if not hpc_file.exists():
        if action in (FileCheckType.Exist, FileCheckType.Executable, FileCheckType.IsDir):
            raise SubError(f'{value} does not exist')
        elif action == FileCheckType.Remind:
            logger.warning(f'{value} does not exist')
        elif action == FileCheckType.MakeDir:
            hpc_file.mkdir()
    if action == FileCheckType.Executable and not hpc_file.is_executable():
        raise SubError(f'{value} is not executable')
    elif action == FileCheckType.IsDir and not hpc_file.is_dir():
        raise SubError(f'{value} is not directory')
    return hpc_file


class JobTimeCheck(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            h, m = values.split(':')
            assert 0 <= int(h) <= 23
            assert 0 <= int(m) <= 60
            setattr(namespace, self.dest, f'{h:02}{m:02}')
        except Exception as e:
            parser.error(f'invalid format of time: {str(e)}')


class SubParser:
    """
    main parser of sub
    """
    _all_apps: Dict[str, "SubParser"] = {}
    server_info = ServerInfo()
    queue_info = Queues()
    logger = logger
    user_groups = groups
    user = user

    def __init_subclass__(cls, **kwargs):
        """
        register subclass
        """
        super().__init_subclass__(**kwargs)
        app = cls.__name__.lower()
        cls.__app__ = app
        script = HPCFile(os.path.join(script_root, 'run_scripts', f'{app}.sh'))
        if script.is_file():
            cls.__entry__ = script
        cls._all_apps[app] = cls
        cls.app_conf: AppModel = all_app_config.get(app) or all_app_config['custom']

    def __init__(self):
        self.parser = ArgumentParser(description=f"This script is used for HPC job submission, version: {__version__}",
                                     epilog=f'if you have any question or suggestion, please contact with {__author__}')
        self.parser.add_argument('-n', '--name', help='job name')
        self.parser.add_argument('-l', '--log_level', choices=['error', 'info', 'debug'], default='info',
                                 help='logging level')
        self.parser.add_argument('-F', '--format', choices=['json', 'raw'], help='std output format')
        self.parser.add_argument('-W', '--wait', nargs='+', metavar='JIDs',
                                 help='start job after those jobs are all succeed')
        self.parser.add_argument('-A', '--after', nargs='+', metavar='JIDs',
                                 help='start job after those jobs are all finished')
        self.parser.add_argument('-p', '--priority', type=int, choices=range(-1024, 1024), metavar='-1024-1023',
                                 help='job priority')
        self.parser.add_argument('-P', '--project', help='name of your project')
        self.parser.add_argument('-a', '--at', metavar='hh:mm', action=JobTimeCheck,
                                 help='time after which the job is eligible for execution')
        self.parser.add_argument('-t', '--time_cost', type=int, help='how many hours will cost, unit: hour')
        self.parser.add_argument('--keep', action='store_true', help='keep running until job done')
        self.parser.add_argument('--hold', action='store_true', help='hold job after submission')
        self.parser.add_argument('--no_mail', action='store_true',
                                 help='disable email notification, or "export SUB_NO_MAIL=YES"')
        self.parser.add_argument('--show_app', action='store_true', help='show application detail')
        app = self.parser.add_subparsers(dest='app', help='application list')
        for soft, cls in sorted(self._all_apps.items()):
            description = cls.app_conf.Description or soft.capitalize()
            parser = app.add_parser(soft, help=description)
            cls.add_parser(parser)

    @staticmethod
    def file_argument_factory(action: FileCheckType = FileCheckType.Exist, formats: Optional[List[str]] = None):
        class FileArgument(Action):
            def __call__(self, parser, namespace, values, option_string=None):
                try:
                    if isinstance(values, list):
                        realpath = []
                        for p in values:
                            if p.endswith('.lst'):  # parse list file
                                try:
                                    with open(p) as f:
                                        file_list = [x.split(',')[0].strip('\n ') for x in f if x[0] not in '$#']
                                except Exception as e:
                                    parser.error(f'invalid file {p}: error: {str(e)}')
                                realpath.extend(check_file(action, i, formats=formats) for i in file_list)
                            else:
                                realpath.append(check_file(action, p, formats=formats))
                    else:
                        realpath = check_file(action, values, formats=formats)
                    setattr(namespace, self.dest, realpath)
                except SubError as e:
                    parser.error(e.err_msg)

        return FileArgument

    @staticmethod
    def max_value_factory(max_value: int):
        class MaxValueAction(Action):
            def __call__(self, parser, namespace, values, option_string=None):
                if int(values) > max_value:
                    parser.error(f'max {self.dest} is {max_value}')
                setattr(namespace, self.dest, values)

        return MaxValueAction

    @classmethod
    def pipeline_factory(cls, pipeline_type: AppType):
        def parse_action(tokens):
            item = tokens[0]
            if extra_cls := (cls._all_apps.get(f'{cls.__app__}_{item}') or cls._all_apps.get(item)):
                if extra_cls.app_conf.Type == pipeline_type:
                    return extra_cls
            elif item.endswith('.sh'):
                return check_file(action=FileCheckType.Executable, value=item)
            else:
                raise SubError(f'can not find handler for {item}')

        def parse_pipeline(pipeline: str) -> pp.ParseResults:
            serial_op = pp.Literal('->')
            para_op = pp.Suppress(',')
            element = pp.Word(pp.alphanums + './_')
            element.set_parse_action(parse_action)
            grammar = pp.infix_notation(element, [(serial_op, 2, pp.opAssoc.LEFT), (para_op, 2, pp.opAssoc.LEFT)])
            pipeline = grammar.parseString(pipeline, parse_all=True)
            return pipeline

        class PipeLine(Action):
            def __call__(self, parser, namespace, values, option_string=None):
                try:
                    setattr(namespace, self.dest, parse_pipeline(values))
                except pp.ParseException:
                    parser.error(f'{values} contains illegal characters')
                except SubError as e:
                    parser.error(e.err_msg)

        return PipeLine

    @staticmethod
    def fix_jobname(name: str) -> str:
        return re.sub(r'\W', '_', name.rsplit('.', maxsplit=1)[0])

    @classmethod
    def add_custom_parser(cls, parser: "ArgumentParser"):
        pass

    @classmethod
    def add_parser(cls, parser: "ArgumentParser"):
        app = cls.app_conf.Name
        if versions := cls.app_conf.Versions:
            default_version = cls.app_conf.DefaultVersion
            parser.add_argument('-v', '--version', choices=versions,
                                help=f"available versions, default is {default_version or 'auto select'}")
            if default_version:
                parser.set_defaults(version=default_version)
        max_gpu, max_cpu = cls.app_conf.MaxGPU, cls.app_conf.MaxCores
        default_cpu, default_gpu = cls.app_conf.DefaultMinCores, cls.app_conf.DefaultGPU
        if default_cpu > 0:
            cpu_msg = f'{default_cpu}+'
        elif default_cpu == 0:
            cpu_msg = 'GPU only'
            if max_cpu > 0:
                raise ValueError(f'{app} template config error')
        elif default_cpu == -1:
            cpu_msg = 'all cores in HPC node'
        elif default_cpu == -2:
            cpu_msg = 'all cores in one CPU socket'
        elif default_cpu == -3:
            cpu_msg = 'all cores in one numa node'
        else:
            raise ValueError(f'{app} template config error')
        if max_cpu > 0 and max_gpu == 0:  # only support user choose CPU
            parser.add_argument('-c', '--core', type=int, action=cls.max_value_factory(max_cpu),
                                help=f'how many cpu cores you request, default is {cpu_msg}, max is {max_cpu}')
        elif max_cpu <= 0 < max_gpu:  # only support GPU
            parser.add_argument('-g', '--gpu', type=int, action=cls.max_value_factory(max_gpu),
                                help=f'how many gpu you request, default is {default_gpu}, max is {max_gpu}')
        elif max_cpu > 0 and max_gpu > 0:  # support both CPU and GPU
            g = parser.add_mutually_exclusive_group()
            g.add_argument('-c', '--core', type=int, action=cls.max_value_factory(max_cpu),
                           help=f'how many cpu cores you request, default is {cpu_msg}, max is {max_cpu}')
            g.add_argument('-g', '--gpu', type=int, action=cls.max_value_factory(max_gpu),
                           help=f'how many gpu you request, default is {default_gpu}, max is {max_gpu}')
        if valid_queues := [q.Name for q in cls.queue_info.valid_queues(cls.__app__)]:
            parser.add_argument('-q', '--queue', choices=valid_queues, help='PBS queue')
        cls.add_custom_parser(parser)
        parser.add_argument('jobfile', nargs='+', action=cls.file_argument_factory(formats=cls.app_conf.Formats),
                            help='job files')
        parser.add_argument('--pre', action=cls.pipeline_factory(AppType.PreProcessing), metavar='PIPE_LINE',
                            help='pre processing pipeline')
        parser.add_argument('--post', action=cls.pipeline_factory(AppType.PostProcessing), metavar='PIPE_LINE',
                            help='post processing pipeline')
        for n, c in cls._all_apps.items():
            if n.startswith(f'{app}_') and (app_type := c.app_conf.Type) in (
                    AppType.PreProcessing, AppType.PostProcessing):
                sub_parser = parser.add_argument_group(f'{app_type.value} {n[len(app) + 1:]}')
                c.add_custom_parser(sub_parser)
        return parser

    @classmethod
    def set_version(cls, args: "Namespace", qsub: "Qsub"):
        if version := getattr(args, 'version', None):
            qsub.variable['VERSION'] = version

    @classmethod
    def set_script(cls, qsub: "Qsub", script: HPCFile):
        qsub.variable["APP"] = cls.__app__
        qsub.script = script
        if not script.is_executable():
            raise SubError(f'{script} is not executable')
        qsub.remote = bool(script.level)

    @classmethod
    def set_mpi_omp(cls, args: "Namespace", qsub: "Qsub"):
        if (mpi := cls.app_conf.MPI) is not None:
            qsub.mpi = mpi
        qsub.openmp = cls.app_conf.OpenMP

    @classmethod
    def set_queue_cores(cls, args: "Namespace", qsub: "Qsub", main_job=True):
        user_set_cpu = getattr(args, 'core', None) if main_job else None
        user_set_gpu = getattr(args, 'gpu', None) if main_job else None
        queue = getattr(args, 'queue', None) if main_job else None
        job_count = 1
        if (job_files := getattr(args, 'jobfile')) and isinstance(job_files, list):
            job_count = len(job_files)
        cores, gpu_flag, q = cls.queue_info.recommend(cls.app_conf, user_set_cpu, user_set_gpu, queue, job_count)
        qsub.cores = cores
        qsub.gpu = gpu_flag
        qsub.queue = q
        if memory := cls.app_conf.Memory:  # set memory
            qsub.memory = memory

    @classmethod
    def set_custom_option2(cls, args: "Namespace", qsub: "Qsub"):
        pass

    @classmethod
    def set_custom_option1(cls, args: "Namespace", qsub: "Qsub"):
        pass

    @classmethod
    def set_license(cls, args: "Namespace", qsub: "Qsub"):
        if cost := cls.app_conf.LicenseCost:
            lic = cls.app_conf.LicenseName or cls.__app__
            logger.debug(f'setting license {lic} to {cost}')
            setattr(qsub, lic, PbsRes(cost))

    @classmethod
    def add_job_file(cls, qsub: Qsub, job_file: HPCFile):
        qsub.variable['JOBFILE'] = job_file
        if not hasattr(qsub, 'project'):
            project = cls.server_info.guess_project(job_file)
            qsub.project = project
        if not hasattr(qsub, 'jobname'):
            qsub.jobname = cls.fix_jobname(job_file.name)
        if job_file.level != 0:
            qsub.workdir = job_file.absolute() if job_file.is_dir() else job_file.parent
            qsub.output = qsub.workdir
        else:
            raise SubError(f"{job_file} is not in share storage")

    @classmethod
    def set_base_args(cls, args: "Namespace", qsub: "Qsub"):
        """
        set base argument
        """
        if not (args.no_mail or "SUB_NO_MAIL" in os.environ):
            qsub.email = email
        if at := args.at:
            qsub.at = at
        if args.hold:
            qsub.hold = True
        if wait_jid := args.wait:
            qsub.waitfor.extend(wait_jid)
        if after_jid := args.after:
            qsub.after.extend(after_jid)
        if priority := args.priority:
            qsub.priority = priority
        if jobname := args.name:
            qsub.jobname = jobname
        if time_cost := args.time_cost:
            qsub.walltime = WallTime(hour=time_cost)
        if project := args.project:
            qsub.project = project
        cls.set_version(args, qsub)
        if getattr(cls, '__entry__', None):
            cls.set_script(qsub, cls.__entry__)
        cls.set_custom_option1(args, qsub)

    @classmethod
    def handle(cls, args: "Namespace", qsub: "Qsub", main_job=True):
        cls.set_queue_cores(args, qsub, main_job)
        cls.set_mpi_omp(args, qsub)
        cls.set_license(args, qsub)
        cls.set_custom_option2(args, qsub)

    def run_pipeline(self, pipeline_type: AppType,
                     pipeline: pp.ParseResults,
                     args: Namespace,
                     qsub: Qsub,
                     all_jids: List[str],
                     wait_jids: Optional[List[str]] = None,
                     test_info: Optional[list] = None) -> List[Optional[dict]]:
        """
        pipeline_type: pre-processing or post-processing
        pipeline: pyparsing result
        args: parsed arguments
        qsub: Qsub instance
        all_jids: all job IDs will be saved into all_jids
        wait_jids: recursion variable
        test_info: all test info will be saved into test_info
        """
        if wait_jids is None:
            wait_jids = []
        jids = []
        for i in pipeline:
            if i == '->':
                wait_jids, jids = jids, []
            elif isinstance(i, pp.ParseResults):
                jids.extend(self.run_pipeline(pipeline_type, i, args, qsub, all_jids, wait_jids, test_info))
            else:
                with qsub:
                    if isinstance(i, HPCFile):  # user-define pipeline handler
                        cls = self._all_apps['custom']
                        cls.set_script(qsub, i)
                    elif issubclass(i, self.__class__):  # builtin pipeline handler
                        cls = i
                        cls.set_script(qsub, cls.__entry__)
                    else:
                        raise TypeError('pipeline type error')
                    cls.handle(args, qsub, main_job=False)
                    if pipeline_type == AppType.PreProcessing:
                        qsub.jobname = qsub.jobname + '_pre'
                    elif pipeline_type == AppType.PostProcessing:
                        qsub.jobname = qsub.jobname + '_post'
                    jid = qsub.run(wait_jids=wait_jids, test_info=test_info)
                    jids.append(jid)
                    all_jids.append(jid)
        return jids

    def run(self, test=False):
        """
        program entry
        """
        args = self.parser.parse_args()
        log_level = args.log_level
        output_format = args.format or 'raw' if log_level == 'error' else None
        stream_log.setLevel(getattr(logging, log_level.upper()))
        logger.debug(f'pbs data is {api.delay()} seconds ago')
        if not (app := args.app) in self._all_apps:
            self.queue_info.show(show_config=args.show_app)
            return
        jids = []
        debug_details = []
        cls = self._all_apps[app]
        qsub = Qsub()
        qsub.variable['LOG_LEVEL'] = log_level
        test_info = debug_details if test else None
        try:
            cls.set_base_args(args, qsub)
            jobfiles = getattr(args, 'jobfile', [])
            for job_file in jobfiles:
                with qsub:
                    wait_jids = []
                    cls.add_job_file(qsub, job_file)
                    if pre_processing := getattr(args, 'pre', None):
                        wait_jids = self.run_pipeline(AppType.PreProcessing, pre_processing, args, qsub, jids,
                                                      test_info=test_info)
                    with qsub:
                        cls.handle(args, qsub)
                        main_jid = qsub.run(wait_jids=wait_jids, test_info=test_info)
                        jids.append(main_jid)
                    if post_processing := getattr(args, 'post', None):
                        self.run_pipeline(AppType.PostProcessing, post_processing, args, qsub, jids,
                                          test_info=test_info,
                                          wait_jids=[main_jid])
            if output_format == 'json':
                print(json.dumps({'job_ids': jids}, indent=True))
            elif output_format == 'raw':
                for j in jids:
                    print(j)
            while args.keep and (all_jids := Jobs.unfinished_jobs(jids)):
                logger.info(f'job {all_jids} are not finished')
                time.sleep(60)
        except SubError as e:
            logger.error(f'{Fore.RED}{(msg := e.err_msg)}{Style.RESET_ALL}')
            if output_format == 'json':
                print(json.dumps({'message': msg}, indent=True))
            elif output_format == 'raw':
                print(msg)
            return msg
        return debug_details
