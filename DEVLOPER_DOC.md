# 开发者文档

## 如何添加一个新的软件扩展

需要在software目录下放置对应的扩展代码，在代码中，需要

- 从mainparser中引入MainParser，并实现一个子类
- 在子类中定义__software__类变量，值为软件名称， 定义__version__类变量为当前扩展的版本号
- 在子类中实现add_parser类方法，添加自己的子解析器
- 在子类中实现handle类方法，作为回调函数。函数中的args参数为主parse_args后的值，而base_args为主parser预处理后的值。这个handle方法最终返回的是一个pbs任务id的列表。

代码放置到指定位置以后，需要在software/__init__.py 下面手工将新的插件引入到命名空间中，使新的子类生效
