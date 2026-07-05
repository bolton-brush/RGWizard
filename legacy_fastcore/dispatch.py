"""Compatibility subset of fastcore 1.7 dispatch.

Old exported fastai learners can contain pickled references to
fastcore.dispatch.TypeDispatch. fastcore 1.8 moved that API, so we keep the
legacy implementation available only while loading those model files.
"""

from __future__ import annotations

from collections import defaultdict

from fastcore.foundation import *
from fastcore.imports import *
from fastcore.utils import *


__all__ = [
    "typedispatch",
    "lenient_issubclass",
    "sorted_topologically",
    "TypeDispatch",
    "DispatchReg",
    "retain_meta",
    "default_set_meta",
    "cast",
    "retain_type",
    "retain_types",
    "explode_types",
]


def lenient_issubclass(cls, types):
    "If possible return whether `cls` is a subclass of `types`, otherwise return False."
    if cls is object and types is not object:
        return False
    try:
        return isinstance(cls, types) or issubclass(cls, types)
    except Exception:
        return False


def sorted_topologically(iterable, *, cmp=operator.lt, reverse=False):
    "Return a new list containing all items from the iterable sorted topologically"
    items, res = L(list(iterable)), []
    for _ in range(len(items)):
        item = items.reduce(lambda x, y: y if cmp(y, x) else x)
        res.append(item)
        items.remove(item)
    return res[::-1] if reverse else res


def _chk_defaults(f, ann):
    pass


def _p2_anno(f):
    "Get the first two annotations of `f`, defaulting to `object`."
    hints = type_hints(f)
    ann = [o for n, o in hints.items() if n != "return"]
    if callable(f):
        _chk_defaults(f, ann)
    while len(ann) < 2:
        ann.append(object)
    return ann[:2]


class _TypeDict:
    def __init__(self):
        self.d, self.cache = {}, {}

    def _reset(self):
        self.d = {k: self.d[k] for k in sorted_topologically(self.d, cmp=lenient_issubclass)}
        self.cache = {}

    def add(self, t, f):
        "Add type `t` and function `f`."
        if not isinstance(t, tuple):
            t = tuple(L(union2tuple(t)))
        for item in t:
            self.d[item] = f
        self._reset()

    def all_matches(self, k):
        "Find all matching types that are superclasses of `k`."
        if k not in self.cache:
            types = [f for f in self.d if lenient_issubclass(k, f)]
            self.cache[k] = [self.d[o] for o in types]
        return self.cache[k]

    def __getitem__(self, k):
        "Find the first matching type that is a superclass of `k`."
        res = self.all_matches(k)
        return res[0] if len(res) else None

    def __repr__(self):
        return self.d.__repr__()

    def first(self):
        return first(self.d.values())


class TypeDispatch:
    "Dictionary-like object; `__getitem__` matches keys of types using `issubclass`."

    def __init__(self, funcs=(), bases=()):
        self.funcs, self.bases = _TypeDict(), L(bases).filter(is_not(None))
        for o in L(funcs):
            self.add(o)
        self.inst = None
        self.owner = None

    def add(self, f):
        "Add function `f` by its first two type annotations."
        if isinstance(f, staticmethod):
            a0, a1 = _p2_anno(f.__func__)
        else:
            a0, a1 = _p2_anno(f)
        target = self.funcs.d.get(a0)
        if target is None:
            target = _TypeDict()
            self.funcs.add(a0, target)
        target.add(a1, f)

    def first(self):
        "Get first function in ordered dict of type:function."
        return self.funcs.first().first()

    def returns(self, x):
        "Get the return type annotation for `x`."
        return anno_ret(self[type(x)])

    def _attname(self, k):
        return getattr(k, "__name__", str(k))

    def __repr__(self):
        res = [
            f"({self._attname(k)},{self._attname(l)}) -> {getattr(v, '__name__', type(v).__name__)}"
            for k in self.funcs.d
            for l, v in self.funcs[k].d.items()
        ]
        res += [o.__repr__() for o in self.bases]
        return "\n".join(res)

    def __call__(self, *args, **kwargs):
        types = L(args).map(type)[:2]
        f = self[tuple(types)]
        if not f:
            return args[0]
        if isinstance(f, staticmethod):
            f = f.__func__
        elif self.inst is not None:
            f = MethodType(f, self.inst)
        elif self.owner is not None:
            f = MethodType(f, self.owner)
        return f(*args, **kwargs)

    def __get__(self, inst, owner):
        self.inst = inst
        self.owner = owner
        return self

    def __getitem__(self, k):
        "Find first matching type that is a superclass of `k`."
        k = L(k)
        while len(k) < 2:
            k.append(object)
        matches = self.funcs.all_matches(k[0])
        for target in matches:
            res = target[k[1]]
            if res is not None:
                return res
        for base in self.bases:
            res = base[k]
            if res is not None:
                return res
        return None


class DispatchReg:
    "A global registry for `TypeDispatch` objects keyed by function name."

    def __init__(self):
        self.d = defaultdict(TypeDispatch)

    def __call__(self, f):
        if isinstance(f, (classmethod, staticmethod)):
            name = f"{f.__func__.__qualname__}"
        else:
            name = f"{f.__qualname__}"
        if isinstance(f, classmethod):
            f = f.__func__
        self.d[name].add(f)
        return self.d[name]


typedispatch = DispatchReg()


def retain_meta(x, res, as_copy=False):
    "Call `res.set_meta(x)`, if it exists."
    if hasattr(res, "set_meta"):
        res.set_meta(x, as_copy=as_copy)
    return res


def default_set_meta(self, x, as_copy=False):
    "Copy over `_meta` from `x` if it is missing."
    if hasattr(x, "_meta") and not hasattr(self, "_meta"):
        meta = x._meta
        if as_copy:
            meta = copy(meta)
        self._meta = meta
    return self


@typedispatch
def cast(x, typ):
    "Cast `x` to type `typ`."
    res = typ._before_cast(x) if hasattr(typ, "_before_cast") else x
    if risinstance("ndarray", res):
        res = res.view(typ)
    elif hasattr(res, "as_subclass"):
        res = res.as_subclass(typ)
    else:
        try:
            res.__class__ = typ
        except Exception:
            res = typ(res)
    return retain_meta(x, res)


def retain_type(new, old=None, typ=None, as_copy=False):
    "Cast `new` to type of `old` or `typ` if it is a superclass."
    if new is None:
        return
    assert old is not None or typ is not None
    if typ is None:
        if not isinstance(old, type(new)):
            return new
        typ = old if isinstance(old, type) else type(old)
    if typ == NoneType or isinstance(new, typ):
        return new
    return retain_meta(old, cast(new, typ), as_copy=as_copy)


def retain_types(new, old=None, typs=None):
    "Cast each item of `new` to the type of the matching item in `old`."
    if not is_listy(new):
        return retain_type(new, old, typs)
    if typs is not None:
        if isinstance(typs, dict):
            target = first(typs.keys())
            typs = typs[target]
        else:
            target, typs = typs, None
    else:
        target = type(old) if old is not None and isinstance(old, type(new)) else type(new)
    return target(L(new, old, typs).map_zip(retain_types, cycled=True))


def explode_types(o):
    "Return the type of `o`, potentially nested for list-like objects."
    if not is_listy(o):
        return type(o)
    return {type(o): [explode_types(item) for item in o]}

