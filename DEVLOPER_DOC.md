# 开发者文档

改脚本设计的目的是为提供了一个统一的脚本入口，设计包含以下目标

- 提供一个统一的命令入口， 包含所有任务都可以使用的公共参数
- 每个求解器可以有独立的参数和处理过程，代码和其他求解软件独立
- 引擎的核心部分需要提供公共的方法，提供给每个求解脚本引用，让新增加扩展的代码变得最小 

## 如何添加一个新求解器扩展

以acusolv为例来说明

```python
import os

from .mainparser import MainParser


class AcuSolve(MainParser):
    __software__ = 'acusolv'
    __version__ = '1.0.0'

    @classmethod
    def add_parser(cls, parser):
        parser.add_argument('--do', choices=['prep,view,solve,prep-solve,all'], default='all',
                            help='execute modules, default:all')
        super(AcuSolve, cls).add_parser(parser)

    @classmethod
    def handle(cls, args, qsub) -> list:
        if args.queue == 'gpu':
            gpu = True
            raw_cores = 1
        else:
            gpu = False
            raw_cores = args.core
        queue, cores, jobfile = cls.default_policy(args.queue, raw_cores, qsub, maxcores=128, jobfile=args.jobfile,
                                                   gpu=gpu)
        case = os.path.basename(jobfile).rsplit('.', maxsplit=1)[0]
        qsub.variable = f'case={case},cores={cores},do={args.do}'
        jid = qsub.run()
        return [jid]
```

1. 在software目录中，创建comsol.py这个文件， `__init__.py` 所有放置在software下面的.py文件都会被自动加载
2. 在求解器代码中，import MainParser, 定义一个MainParser的子类， 所有MainParser子类都会在`__init_subclass__`的作用下被注册到MainParser中
3. 在子类中需要定义`__software__`, `__version__` 这两个类属性，表示软件的名称和这个脚本的版本
4. 定义add_parser方法加入该求解软件所需要支持的额外参数，大部分求解软件都需要有核心数，队列和jobfile文件， 为了兼容过去的代码，这几个参数被设置成求解软件的参数而非公共参数，可以直接从父类继承
5. 定义handle方法，这个方法作是求解器运行的主要执行逻辑， 
   1. 接受的参数有两个args和qsub, 其中，args是已经被解析好的ArgumentParser的参数， 而qsub则是pbs的qsub命令映射的对象。 在handle方法被执行之前，很多qsub的参数已经被设置好了 
   2. 这个方法的返回值是一个提交好任务的jid列表。 如果一次只提交一个任务，就返回\[jid]即可
   3. 使用cls.default_policy，声明使用默认的调度规则，这个规则描述如下:
      - 在pbs的队列设置中，配置每个队列允许执行的软件和允许使用的团队， 如果queue的参数为None，则根据当前集群的压力以及可选的选择一个空闲cpu核心数最多的队列。 如果queue被指定，则检查这个队列是否是可用的
      - 如果core参数为None，则根据当前队列选择一个默认的核心数， 如果core不为None，则对该核心数进行检查，来确定这个核心数是否合规。合规的核心数属于以下条件中的一个:
        - 核心数大于队列中单台主机的核心数，必须是主机核心数的整数倍，同时小于maxcores这个参数。 如果在设置了mpi=True后，调度会被限制在同一个ib交换机下
        - 核心数小于队列中单台主机的核心数，同时大于每个numa节点的核心数，则核心数必须是单个numa的节点cpu数的整数倍，调度会被约束在最近的numa分组，通常情况下是同一个cpu socket下面
        - 核心数等于一个numa节点的cpu核心数，调度会被限制在这个numa节点的内部
      - 如果jobfile被设置了，那么还会有如下操作:
        - 对jobfile的文件路径进行检查
        - 切换到jobfile的路径下，进行任务提交，并把qsub的输出路径设置成和jobfile的路径一致
        - 根据jobfile的完整路径猜测当前的任务属于哪个项目
        - 如果任务名称没有被指定，则根据jobfile，设置一个任务名称
      - 如果没有指定script参数， 实际执行的脚本被设置成run_scripts下面软件名.sh的脚本
   4. 设置脚本需要的其他环境变量。 所有的脚本变量，都是通过环境变量的方式进行传递
   5. 执行qsub.run()， 这个会将所有的参数转换成qsub命令进行执行，并返回任务的jid。 数据库记录的功能也在这里面实现
6. 在run_scripts中放置软件实际执行的求解器模版, 默认名称为软件名.sh, 建议使用-e来启动，让pbs捕获脚本中的异常≈

```bash
#!/bin/bash -e

cd ${PBS_O_WORKDIR}

if [[ -c /dev/nvidiactl ]]; then
    gpu_flag=-gpu
fi

source /home/hpcsw/hwCFD2021/altair/hwcfdsolvers/acusolve/linux64/script/acusim.sh
export ALTAIR_LICENSE_PATH=xxxxx
acuRun -pb $case -np $cores -do $do $gpu_flag &> ${PBS_JOBID}.log
```
