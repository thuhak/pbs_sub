# author: thuhak.zhou@nio.com
from contextlib import contextmanager
import enum
import os

from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Enum
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

dbhost = os.environ.get('PBS_SUB_DB')
dbpass = os.environ.get('PBS_SUB_DB_PW')

engine_path = f'mysql+pymysql://hpc:{dbpass}@{dbhost}/hpc'
engine = create_engine(engine_path, encoding='utf-8', echo=False,
                       pool_size=10, max_overflow=10, pool_recycle=7200, pool_pre_ping=True)
Base = declarative_base()


@contextmanager
def session_scope():
    session = sessionmaker(bind=engine)()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


class JobStat(enum.Enum):
    """
    job state array, use `man qstat` to see detail.
    """
    Unknown = 0
    B = 1
    E = 2
    F = 3
    H = 4
    M = 5
    Q = 6
    R = 7
    S = 8
    T = 9
    U = 10
    W = 11
    X = 12


class Job(Base):
    """
    PBS job table
    """
    __tablename__ = 'job'
    jid = Column(Integer, primary_key=True, autoincrement=False, comment='pbs job id')
    software = Column(String(64), index=True, comment='software name')
    module = Column(String(1024), server_default='main', comment='software module')
    user = Column(String(64), comment='submit user')
    jobfile = Column(String(1024), comment='job file')
    project = Column(String(128), comment='project')
    cores = Column(Integer, index=True, comment='cpu cores')
    queue = Column(String(32), comment='pbs queue')
    stime = Column(DateTime, comment='job start time')
    walltime = Column(Integer, comment='job cost time, unit: second')
    state = Column(Enum(JobStat), default=JobStat.Unknown, comment='job state')
    license_server = Column(String(1024), comment='license server')
    extra = Column(JSON, comment='extra data, json format')


if __name__ == '__main__':
    Base.metadata.create_all(engine)
