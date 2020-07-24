# PBS任务提交脚本

为PBS任务提交提供一个统一的入口和可扩展框架。
脚本主要实现以下功能特性:

- 参数检查，让用户能够在任务提交前前发现自己的输入参数问题。防止发生一些意想不到的情况
- 许可服务器检查以及选择
- 更智能的资源分配，可以根据当前的队列使用情况动态扩展申请的资源量
- 若干小工具，比如查询队列空闲cpu等。更易于使用

## 如何使用

这是一个纯命令行工具。参数分为两个层次：

- 公共参数：在所有的软件中都有效
- 私有参数：只在对应的软件扩展下生效

大致格式为./pbs_sub.py [公共参数] 软件名称 [私有参数]

可以使用
```bash
./pbs_sub.py
```
打印所有的公共参数说明，以及支持的软件列表和对应的脚本版本, 以及一些工具命令。

例如

```bash
./pbs_sub.py --free_cores
```
打印出当前pbs各个队列的剩余核心数

使用
```bash
./pbs_sub.py SOFTWARE -h
```
打印对应的软件的的使用说明