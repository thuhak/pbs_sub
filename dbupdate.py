#!/usr/bin/env python3.6
# author: thuhak.zhou@nio.com
from software.database import session_scope, Job, JobStat
import json
import sh
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def get_info(jid):
    try:
        QSTAT = sh.Command('/opt/pbs/bin/qstat')
        raw_data = sh.grep(QSTAT('-fx', jid, '-F', 'json'), '-v', 'Submit_arguments').stdout
        job_data = list(json.loads(raw_data)['Jobs'].values())[0]
        return job_data
    except:
        logging.error(f'unable get data for {jid}')
        return {}


def calc_walltime(walltime):
    h, m, s = [int(x) for x in walltime.split(':')]
    return h * 3600 + m * 60 + s


if __name__ == '__main__':
    with session_scope() as sess:
        unfinished_jobs = sess.query(Job).filter(Job.state.notin_([JobStat.E, JobStat.F])).all()
        for job in unfinished_jobs:
            jid = job.jid
            pbs_info = get_info(job.jid)
            if pbs_info:
                logging.info(f'updating state for {jid}')
                try:
                    walltime = calc_walltime(pbs_info['resources_used']['walltime'])
                    job.walltime = walltime
                except:
                    logging.info('can not get walltime')
                try:
                    raw_stime = pbs_info.get('stime') or pbs_info.get('ctime')
                    stime = datetime.strptime(raw_stime, '%a %b  %d %H:%M:%S %Y')
                    job.stime = stime
                except:
                    logging.info('can not get stime')
                try:
                    state = pbs_info['job_state']
                    job.state = state
                except:
                    logging.info('can not get state')

