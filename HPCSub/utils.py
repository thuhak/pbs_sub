"""
utilities
"""
# author: thuhak.zhou@nio.com
from copy import deepcopy
from functools import wraps


__all__ = ['Context', 'singleton']


class Context:
    """
    context decorator
    """

    def __init__(self, cls):
        self._cls = cls
        self._stack = []

    def __call__(self, *args, **kwargs):
        self._package = self._cls(*args, **kwargs)
        return self

    def __setattr__(self, key, value):
        if not key.startswith('_'):
            setattr(self._package, key, value)
        else:
            super(Context, self).__setattr__(key, value)

    def __delattr__(self, item):
        if not item.startswith('_'):
            delattr(self._package, item)
        else:
            super(Context, self).__delattr__(item)

    def __getattr__(self, item):
        return getattr(self._package, item)

    def __enter__(self):
        self._stack.append(self._package)
        self._package = deepcopy(self._package)  # TODO: use copy-on-write instead of deepcopy

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._package = self._stack.pop()


def singleton(cls):
    """
    singleton decorator
    """
    instances = {}

    @wraps(cls)
    def instance(*args, **kwargs):
        return instances.setdefault(cls, cls(*args, **kwargs))

    return instance
