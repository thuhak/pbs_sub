# author: thuhak.zhou@nio.com
import os
from ..mainparser import SubParser, HPCFile, WallTime


class Custom(SubParser):
    """
    custom script wrapper
    """

    @classmethod
    def add_job_file(cls, qsub, job_file: HPCFile):
        if not hasattr(qsub, 'jobname'):
            qsub.jobname = cls.fix_jobname(job_file.name)
        cls.set_script(qsub, job_file)

    @classmethod
    def set_custom_option2(cls, args, qsub):
        title = qsub.script.name.rsplit('.', maxsplit=1)[0].upper() + '_'
        qsub.variable.update({k: v for k, v in os.environ.items() if k.startswith(title)})
        if not getattr(qsub, 'walltime', None):
            walltime = WallTime()
            if (cores := qsub.cores) == 1:
                walltime.hour = 8
            elif 1 < cores <= 8:
                walltime.hour = 4
            else:
                walltime.hour = 2
            qsub.walltime = walltime

