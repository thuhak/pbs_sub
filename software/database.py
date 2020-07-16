# author: thuhak.zhou@nio.com
from contextlib import contextmanager
from collections import Iterable
import enum

from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON, Enum
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


# add database config here
dbconfig = {'user': 'USER', 'pass': 'PASSWORD', 'host': 'HOST', 'port': 'PORT', 'db': 'DB'}

engine_path = 'mysql+pymysql://{user}:{pass}@{host}:{port}/{db}'.format(**dbconfig)
Base = declarative_base()
engine = create_engine(engine_path, encoding='utf-8', echo=False,
                       pool_size=10, max_overflow=10, pool_recycle=7200, pool_pre_ping=True)
DBSession = sessionmaker(bind=engine)


@contextmanager
def session_scope():
    session = DBSession()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


class BaseModel(Base):
    __abstract__ = True
    hide = set()
    other_prop = set()

    def to_dict(self, rel=True, hide=True) -> dict:
        if hide:
            res = {column.key: getattr(self, attr)
                   for attr, column in self.__mapper__.c.items() if attr not in self.hide}
        else:
            res = {column.key: getattr(self, attr)
                   for attr, column in self.__mapper__.c.items()}
        for prop in self.other_prop:
            if hide and prop in self.hide:
                continue
            res[prop] = getattr(self, prop)
        if rel:
            for attr, relation in self.__mapper__.relationships.items():
                item = getattr(self, attr)
                if isinstance(item, Iterable):
                    res[attr] = [i.to_dict(rel=False) for i in item]
                else:
                    res[attr] = item.to_dict(rel=False)
        return res


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


class Job(BaseModel):
    """
    PBS job table
    """
    __tablename__ = 'job'

    jid = Column(Integer, primary_key=True, autoincrement=False, comment='pbs job id')
    software = Column(String(64), index=True, comment='software name')
    module = Column(String(1024), server_default='main', comment='software module')
    user = Column(String(64), comment='submit user')
    jobfile = Column(String(1024), comment='job file')
    cores = Column(Integer, index=True, comment='cpu cores')
    queue = Column(String(32), comment='pbs queue')
    stime = Column(DateTime, comment='job start time')
    walltime = Column(Integer, comment='job cost time, unit: second')
    state = Column(Enum(JobStat), default=JobStat.Unknown, comment='job state')
    license_server = Column(String(1024), comment='license server')
    extra = Column(JSON, comment='extra data, json format')


# Base.metadata.create_all(engine)
