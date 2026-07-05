"""Compatibility subset of fastcore 1.7 transform.

This is used only for loading old fastai learner exports that pickle
fastcore.transform.Pipeline/Transform objects.
"""

from fastcore.foundation import *
from fastcore.imports import *
from fastcore.utils import *

from .dispatch import *

import inspect


__all__ = [
    "Sig",
    "Transform",
    "InplaceTransform",
    "DisplayedTransform",
    "ItemTransform",
    "get_func",
    "Func",
    "compose_tfms",
    "mk_transform",
    "gather_attrs",
    "gather_attr_names",
    "Pipeline",
]


_tfm_methods = "encodes", "decodes", "setups"


def _is_tfm_method(n, f):
    return n in _tfm_methods and callable(f)


class _TfmDict(dict):
    def __setitem__(self, k, v):
        if not _is_tfm_method(k, v):
            return super().__setitem__(k, v)
        if k not in self:
            super().__setitem__(k, TypeDispatch())
        self[k].add(v)


class _TfmMeta(type):
    def __new__(cls, name, bases, namespace):
        res = super().__new__(cls, name, bases, namespace)
        for nm in _tfm_methods:
            base_td = [getattr(b, nm, None) for b in bases]
            if nm in res.__dict__:
                getattr(res, nm).bases = base_td
            else:
                setattr(res, nm, TypeDispatch(bases=base_td))
        res.__signature__ = inspect.signature(res.__init__)
        return res

    def __call__(cls, *args, **kwargs):
        f = first(args)
        name = getattr(f, "__name__", None)
        if _is_tfm_method(name, f):
            getattr(cls, name).add(f)
            return f
        obj = super().__call__(*args, **kwargs)
        if hasattr(obj, "__call__"):
            obj.__signature__ = inspect.signature(obj.__call__)
        return obj

    @classmethod
    def __prepare__(cls, name, bases):
        return _TfmDict()


def _get_name(o):
    if hasattr(o, "__qualname__"):
        return o.__qualname__
    if hasattr(o, "__name__"):
        return o.__name__
    return o.__class__.__name__


def _is_tuple(o):
    return isinstance(o, tuple) and not hasattr(o, "_fields")


class Transform(metaclass=_TfmMeta):
    "Delegates `__call__`, `decode`, and `setup` to encodes/decodes/setups."

    split_idx, init_enc, order, train_setup = None, None, 0, None

    def __init__(self, enc=None, dec=None, split_idx=None, order=None):
        self.split_idx = ifnone(split_idx, self.split_idx)
        if order is not None:
            self.order = order
        self.init_enc = enc or dec
        if not self.init_enc:
            return

        self.encodes, self.decodes, self.setups = TypeDispatch(), TypeDispatch(), TypeDispatch()
        if enc:
            self.encodes.add(enc)
            self.order = getattr(enc, "order", self.order)
            if len(type_hints(enc)) > 0:
                self.input_types = union2tuple(first(type_hints(enc).values()))
            self._name = _get_name(enc)
        if dec:
            self.decodes.add(dec)

    @property
    def name(self):
        return getattr(self, "_name", _get_name(self))

    def __call__(self, x, **kwargs):
        return self._call("encodes", x, **kwargs)

    def decode(self, x, **kwargs):
        return self._call("decodes", x, **kwargs)

    def __repr__(self):
        return f"{self.name}:\nencodes: {self.encodes}decodes: {self.decodes}"

    def setup(self, items=None, train_setup=False):
        train_setup = train_setup if self.train_setup is None else self.train_setup
        return self.setups(getattr(items, "train", items) if train_setup else items)

    def _call(self, fn, x, split_idx=None, **kwargs):
        if split_idx != self.split_idx and self.split_idx is not None:
            return x
        return self._do_call(getattr(self, fn), x, **kwargs)

    def _do_call(self, f, x, **kwargs):
        if not _is_tuple(x):
            if f is None:
                return x
            ret = f.returns(x) if hasattr(f, "returns") else None
            return retain_type(f(x, **kwargs), x, ret)
        res = tuple(self._do_call(f, item, **kwargs) for item in x)
        return retain_type(res, x)


add_docs(
    Transform,
    decode="Delegate to decodes to undo transform",
    setup="Delegate to setups to set up transform",
)


class InplaceTransform(Transform):
    "A `Transform` that modifies in-place and just returns what it is passed."

    def _call(self, fn, x, split_idx=None, **kwargs):
        super()._call(fn, x, split_idx, **kwargs)
        return x


class DisplayedTransform(Transform):
    "A transform with a `__repr__` that shows its attrs."

    @property
    def name(self):
        return f"{super().name} -- {getattr(self, '__stored_args__', {})}"


class ItemTransform(Transform):
    "A transform that always takes tuples as items."

    _retain = True

    def __call__(self, x, **kwargs):
        return self._call1(x, "__call__", **kwargs)

    def decode(self, x, **kwargs):
        return self._call1(x, "decode", **kwargs)

    def _call1(self, x, name, **kwargs):
        if not _is_tuple(x):
            return getattr(super(), name)(x, **kwargs)
        y = getattr(super(), name)(list(x), **kwargs)
        if not self._retain:
            return y
        if is_listy(y) and not isinstance(y, tuple):
            y = tuple(y)
        return retain_type(y, x)


def get_func(t, name, *args, **kwargs):
    "Get `t.name`, potentially partialized, or `noop` if not defined."
    f = nested_callable(t, name)
    return f if not (args or kwargs) else partial(f, *args, **kwargs)


class Func:
    "Basic wrapper around a name with args and kwargs to call on a given type."

    def __init__(self, name, *args, **kwargs):
        self.name, self.args, self.kwargs = name, args, kwargs

    def __repr__(self):
        return f"sig: {self.name}({self.args}, {self.kwargs})"

    def _get(self, t):
        return get_func(t, self.name, *self.args, **self.kwargs)

    def __call__(self, t):
        return mapped(self._get, t)


class _Sig:
    def __getattr__(self, k):
        def _inner(*args, **kwargs):
            return Func(k, *args, **kwargs)

        return _inner


Sig = _Sig()


def compose_tfms(x, tfms, is_enc=True, reverse=False, **kwargs):
    "Apply all transforms on `x`, maybe in reverse order."
    if reverse:
        tfms = reversed(tfms)
    for f in tfms:
        if not is_enc:
            f = f.decode
        x = f(x, **kwargs)
    return x


def mk_transform(f):
    "Convert function `f` to `Transform` if it is not already one."
    f = instantiate(f)
    return f if isinstance(f, (Transform, Pipeline)) else Transform(f)


def gather_attrs(o, k, nm):
    "Used in __getattr__ to collect attrs from `self.{nm}`."
    if k.startswith("_") or k == nm:
        raise AttributeError(k)
    att = getattr(o, nm)
    res = [t for t in att.attrgot(k) if t is not None]
    if not res:
        raise AttributeError(k)
    return res[0] if len(res) == 1 else L(res)


def gather_attr_names(o, nm):
    "Used in __dir__ to collect attrs from `self.{nm}`."
    return L(getattr(o, nm)).map(dir).concat().unique()


class Pipeline:
    "A pipeline of composed transforms."

    def __init__(self, funcs=None, split_idx=None):
        self.split_idx, self.default = split_idx, None
        if funcs is None:
            funcs = []
        if isinstance(funcs, Pipeline):
            self.fs = funcs.fs
        else:
            if isinstance(funcs, Transform):
                funcs = [funcs]
            self.fs = L(ifnone(funcs, [noop])).map(mk_transform).sorted(key="order")
        for f in self.fs:
            name = camel2snake(type(f).__name__)
            existing = getattr(self, name, None)
            if existing is not None:
                f = L(existing) + f
            setattr(self, name, f)

    def setup(self, items=None, train_setup=False):
        tfms = self.fs[:]
        self.fs.clear()
        for t in tfms:
            self.add(t, items, train_setup)

    def add(self, ts, items=None, train_setup=False):
        if not is_listy(ts):
            ts = [ts]
        for t in ts:
            t.setup(items, train_setup)
        self.fs += ts
        self.fs = self.fs.sorted(key="order")

    def __call__(self, o):
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)

    def __repr__(self):
        return f"Pipeline: {' -> '.join([f.name for f in self.fs if f.name != 'noop'])}"

    def __getitem__(self, i):
        return self.fs[i]

    def __setstate__(self, data):
        self.__dict__.update(data)

    def __getattr__(self, k):
        return gather_attrs(self, k, "fs")

    def __dir__(self):
        return super().__dir__() + gather_attr_names(self, "fs")

    def decode(self, o, full=True):
        if full:
            return compose_tfms(o, tfms=self.fs, is_enc=False, reverse=True, split_idx=self.split_idx)
        for f in reversed(self.fs):
            if self._is_showable(o):
                return o
            o = f.decode(o, split_idx=self.split_idx)
        return o

    def show(self, o, ctx=None, **kwargs):
        o = self.decode(o, full=False)
        items = (o,) if not _is_tuple(o) else o
        if hasattr(o, "show"):
            ctx = o.show(ctx=ctx, **kwargs)
        else:
            for item in items:
                if hasattr(item, "show"):
                    ctx = item.show(ctx=ctx, **kwargs)
        return ctx

    def _is_showable(self, o):
        if hasattr(o, "show"):
            return True
        if _is_tuple(o):
            return all(hasattr(item, "show") for item in o)
        return False

