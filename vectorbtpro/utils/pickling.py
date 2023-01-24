# Copyright (c) 2021 Oleg Polakow. All rights reserved.

"""Utilities for pickling."""

import attr
import humanize
import ast
from pathlib import Path

import numpy as np
import pandas as pd

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils.path_ import check_mkdir
from vectorbtpro.utils.eval_ import multiline_eval
from vectorbtpro.utils.checks import Comparable, is_hashable, is_deep_equal
from vectorbtpro.utils.formatting import Prettified, prettify_dict

PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


def dumps(obj: tp.Any, **kwargs) -> bytes:
    """Pickle an object to a byte stream."""
    from vectorbtpro.utils.module_ import warn_cannot_import

    warn_cannot_import("dill")
    try:
        import dill as pickle
    except ImportError:
        import pickle

    return pickle.dumps(obj, **kwargs)


def loads(bytes_: bytes, **kwargs) -> tp.Any:
    """Unpickle an object from a byte stream."""
    from vectorbtpro.utils.module_ import warn_cannot_import

    warn_cannot_import("dill")
    try:
        import dill as pickle
    except ImportError:
        import pickle

    return pickle.loads(bytes_, **kwargs)


def save(obj: object, path: tp.Optional[tp.PathLike] = None, mkdir_kwargs: tp.KwargsLike = None, **kwargs) -> None:
    """Pickle an object to a byte stream and write to a file."""
    bytes_ = dumps(obj, **kwargs)
    if path is None:
        path = type(obj).__name__
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".pickle")
    if mkdir_kwargs is None:
        mkdir_kwargs = {}
    check_mkdir(path.parent, **mkdir_kwargs)
    with open(path, "wb") as f:
        f.write(bytes_)


def load(path: tp.PathLike, **kwargs) -> object:
    """Read a byte stream from a file and unpickle."""
    path = Path(path)
    if path.suffix == "" and not path.exists():
        if path.with_suffix(".pickle").exists():
            path = path.with_suffix(".pickle")
        elif path.with_suffix(".pkl").exists():
            path = path.with_suffix(".pkl")
    with open(path, "rb") as f:
        bytes_ = f.read()
    return loads(bytes_, **kwargs)


@attr.s(frozen=True)
class RecState:
    """Class that represents a state used to reconstruct an instance."""

    init_args: tp.Args = attr.ib(default=attr.Factory(tuple))
    """Positional arguments used in initialization."""

    init_kwargs: tp.Kwargs = attr.ib(default=attr.Factory(dict))
    """Keyword arguments used in initialization."""

    attr_dct: tp.Kwargs = attr.ib(default=attr.Factory(dict))
    """Dictionary with names and values of writeable attributes."""


@attr.s(frozen=True)
class RecInfo:
    """Class that represents information needed to reconstruct an instance."""

    id_: str = attr.ib()
    """Identifier."""

    cls: tp.Type = attr.ib()
    """Class."""

    modify_state: tp.Optional[tp.Callable[[RecState], RecState]] = attr.ib(default=None)
    """Callback to modify the reconstruction state."""

    def register(self) -> None:
        """Register self in `rec_info_registry`."""
        rec_info_registry[self.id_] = self


rec_info_registry = {}
"""Registry with instances of `RecInfo` keyed by `RecInfo.id_`.

Populate with the required information if any instance cannot be unpickled."""


def to_class_id(obj: tp.Any) -> tp.Optional[str]:
    """Get the class id from a class.

    If the object is an instance or a subclass of `Pickleable` and `Pickleable._rec_id` is not None,
    uses the reconstruction id. Otherwise, returns the path to the class definition
    with `vectorbtpro.utils.module_.find_class`."""
    from vectorbtpro.utils.module_ import find_class

    if isinstance(obj, type):
        cls = obj
    else:
        cls = type(obj)
    if issubclass(cls, Pickleable):
        if cls._rec_id is not None:
            if not isinstance(cls._rec_id, str):
                raise TypeError(f"Reconstructing id of class {cls} must be a string")
            return cls._rec_id
    class_path = cls.__module__ + "." + cls.__name__
    if find_class(class_path) is not None:
        return class_path
    return None


def from_class_id(class_id: str) -> tp.Optional[tp.Type]:
    """Get the class from a class id."""
    from vectorbtpro.utils.module_ import find_class

    if class_id in rec_info_registry:
        return rec_info_registry[class_id].cls
    cls = find_class(class_id)
    if cls is not None:
        return cls
    raise ValueError(f"Please register an instance of RecInfo for '{class_id}'")


def reconstruct(cls: tp.Union[tp.Hashable, tp.Type], rec_state: RecState) -> object:
    """Reconstruct an instance using a class and a reconstruction state."""
    from vectorbtpro.utils.module_ import find_class

    found_rec = False
    if not isinstance(cls, type):
        class_id = cls
        if class_id in rec_info_registry:
            found_rec = True
            cls = rec_info_registry[class_id].cls
            modify_state = rec_info_registry[class_id].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)
    if not isinstance(cls, type):
        if isinstance(cls, str):
            cls_name = cls
            cls = find_class(cls_name)
            if cls is None:
                cls = cls_name
    if not isinstance(cls, type):
        raise ValueError(f"Please register an instance of RecInfo for '{cls}'")
    if not found_rec:
        class_path = type(cls).__module__ + "." + type(cls).__name__
        if class_path in rec_info_registry:
            cls = rec_info_registry[class_path].cls
            modify_state = rec_info_registry[class_path].modify_state
            if modify_state is not None:
                rec_state = modify_state(rec_state)
    obj = cls(*rec_state.init_args, **rec_state.init_kwargs)
    for k, v in rec_state.attr_dct.items():
        setattr(obj, k, v)
    return obj


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes.

    If any subclass cannot be pickled, override the `Pickleable.rec_state` property
    to return an instance of `RecState` to be used in reconstruction. If the class definition cannot
    be pickled (e.g., created dynamically), override its `_rec_id` with an arbitrary id string,
    dump/save the class, and before loading, map this id to the class in `rec_id_map`. This will use the
    mapped class to construct a new instance."""

    _rec_id: tp.ClassVar[tp.Optional[str]] = None
    """Reconstruction id."""

    def dumps(self, rec_state_only: bool = False, **kwargs) -> bytes:
        """Pickle the instance to a byte stream.

        Optionally, you can set `rec_state_only` to True if the instance will be later unpickled
        directly by the class."""
        if rec_state_only:
            rec_state = self.rec_state
            if rec_state is None:
                raise ValueError("Reconstruction state is None")
            return dumps(rec_state, **kwargs)
        return dumps(self, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableT], bytes_: bytes, **kwargs) -> PickleableT:
        """Unpickle an instance from a byte stream."""
        obj = loads(bytes_, **kwargs)
        if isinstance(obj, RecState):
            obj = reconstruct(cls, obj)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object must be an instance of {cls}")
        return obj

    def encode_config_node(self, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Encode a config node."""
        return value

    @classmethod
    def decode_config_node(cls, key: str, value: tp.Any, **kwargs) -> tp.Any:
        """Decode a config node."""
        return value

    def encode_config(
        self,
        unpack_objects: bool = True,
        compress_unpacked: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        nested: bool = True,
        to_dict: bool = False,
        parser_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Encode the instance to a config string.

        Based on `Pickleable.rec_state`. Raises an error if None.

        Encodes to a format that is guaranteed to be parsed using `Pickleable.decode_config`.
        Otherwise, an error will be thrown. If any object cannot be represented using a string,
        uses `dumps` to convert it to a byte stream.

        When `unpack_objects` is True and an object is an instance of `Pickleable`, saves its
        reconstruction state to a separate section rather than the byte stream. Appends `@` and
        class name to the section name. If `compress_unpacked` is True, will hide keys in
        `RecState` that have empty values. Keys in `RecState` will be appended with `~` to avoid
        collision with user-defined keys having the same name.

        If `use_refs` is True, out of unhashable objects sharing the same id, only the first one
        will be defined while others will store the reference (`&` + key path) to the first one.

        !!! note
            The initial order of keys can be preserved only by using references.

        If `use_class_ids` is True, substitutes any class defined as a value by its id instead of
        pickling its definition. If `to_class_id` returns None, will pickle the definition.

        If the instance is nested, set `nested` to True to represent each sub-dict as a section.

        Other keyword arguments are forwarded to `Pickleable.encode_config_node`."""
        import configparser
        from io import StringIO

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        def _is_dict(dct, _to_dict=to_dict):
            if _to_dict:
                return isinstance(dct, dict)
            return type(dct) is dict

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        def _is_referable(k):
            if "@" in k:
                return False
            if k.endswith("~"):
                return False
            return True

        # Flatten nested dicts
        stack = [(None, "top", self)]
        dct = dict()
        id_paths = dict()
        id_objs = dict()
        while stack:
            parent_k, k, v = stack.pop(0)
            if not isinstance(k, str):
                raise TypeError("Dictionary keys must be strings")
            if parent_k is not None and use_refs and _is_referable(k):
                if id(v) in id_paths:
                    v = "&" + id_paths[id(v)]
                else:
                    if not is_hashable(v):
                        id_paths[id(v)] = _get_path(parent_k) + "." + _get_path(k)
                        id_objs[id(v)] = v  # keep object alive
            if _is_dict(v) and nested:
                if parent_k is not None and use_refs:
                    if parent_k is None:
                        ref_k = _get_path(k)
                    else:
                        ref_k = _get_path(parent_k) + "." + _get_path(k)
                    dct[parent_k][_get_path(k)] = "&" + ref_k
                if parent_k is None:
                    _k = k
                else:
                    _k = _get_path(parent_k) + "." + k
                dct[_k] = dict()
                if len(v) == 0:
                    v = {"_": "_"}
                i = 0
                for k2, v2 in v.items():
                    stack.insert(i, (_k, k2, v2))
                    i += 1
            else:
                if (unpack_objects or k == "top") and isinstance(v, Pickleable):
                    class_id = to_class_id(v)
                    if class_id is None:
                        raise ValueError(f"Class {type(v)} cannot be found. Set reconstruction id.")
                    rec_state = v.rec_state
                    if rec_state is None:
                        if parent_k is None:
                            _k = _get_path(k)
                        else:
                            _k = _get_path(parent_k) + "." + _get_path(k)
                        raise ValueError(f"Must define reconstruction state for '{_k}'")
                    new_v = vars(rec_state)
                    if compress_unpacked and (len(new_v["init_args"]) == 0 and len(new_v["attr_dct"]) == 0):
                        new_v = new_v["init_kwargs"]
                    else:
                        new_v = {k + "~": v for k, v in new_v.items()}
                    stack.insert(0, (parent_k, k + " @" + class_id, new_v))
                else:
                    if parent_k is None:
                        dct[k] = v
                    else:
                        dct[parent_k][k] = v

        # Format config
        for k, v in dct.items():
            parser.add_section(k)
            if len(v) == 0:
                v = {"_": "_"}
            for k2, v2 in v.items():
                v2 = self.encode_config_node(k2, v2, **kwargs)
                if isinstance(v2, str):
                    if not (k2 == "_" and v2 == "_") and not v2.startswith("&"):
                        v2 = repr(v2)
                elif use_class_ids and isinstance(v2, type):
                    class_id = to_class_id(v2)
                    if class_id is not None:
                        v2 = "@" + class_id
                elif isinstance(v2, float) and np.isnan(v2):
                    v2 = "np.nan"
                elif isinstance(v2, float) and np.isposinf(v2):
                    v2 = "np.inf"
                elif isinstance(v2, float) and np.isneginf(v2):
                    v2 = "-np.inf"
                else:
                    try:
                        ast.literal_eval(repr(v2))
                        v2 = repr(v2)
                    except Exception as e:
                        try:
                            float(repr(v2))
                            v2 = repr(v2)
                        except Exception as e:
                            v2 = "!vbt.loads(" + repr(dumps(v2)) + ")"
                parser.set(k, k2, v2)
        with StringIO() as f:
            parser.write(f)
            str_ = f.getvalue()
        return str_

    @classmethod
    def decode_config(
        cls: tp.Type[PickleableT],
        str_: str,
        parse_literals: bool = True,
        run_code: bool = True,
        pack_objects: bool = True,
        use_refs: bool = True,
        use_class_ids: bool = True,
        code_context: tp.KwargsLike = None,
        parser_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> PickleableT:
        """Decode an instance from a config string.

        Can parse configs without sections. Sections can also become sub-dicts if their names use
        the dot notation. For example, section `a.b` will become a sub-dict of the section `a`
        and section `a.b.c` will become a sub-dict of the section `a.b`. You don't have to define
        the section `a` explicitly, it will automatically become the outermost key.

        If a section contains only one pair `_ = _`, it will become an empty dict.

        If `parse_literals` is True, will detect any Python literals and containers such as `True` and `[]`.
        Will also understand `np.nan`, `np.inf`, and `-np.inf`.

        If `run_code` is True, will run any Python code prepended with `!`. Will use the context
        `code_context` together with already defined `np` (NumPy), `pd` (Pandas), and `vbt` (vectorbtpro).

        !!! warning
            Unpickling byte streams and running code has important security implications. Don't attempt
            to parse configs coming from untrusted sources as those can contain malicious code!

        If `pack_objects` is True, will look for class paths prepended with `@` in section names,
        construct an instance of `RecState` (any other keyword arguments will be included to `init_kwargs`),
        and finally use `reconstruct` to reconstruct the unpacked object.

        If `use_refs` is True, will substitute references prepended with `&` for actual objects.
        Constructs a DAG using [graphlib](https://docs.python.org/3/library/graphlib.html).

        If `use_class_ids` is True, will substitute any class ids prepended with `@` with the
        corresponding class.

        Other keyword arguments are forwarded to `Pickleable.decode_config_node`.

        Usage:
            * File `types.ini`:

            ```ini
            string = 'hello world'
            boolean = False
            int = 123
            float = 123.45
            exp_float = 1e-10
            nan = np.nan
            inf = np.inf
            numpy = !np.array([1, 2, 3])
            pandas = !pd.Series([1, 2, 3])
            expression = !dict(sub_dict2=dict(some="value"))
            mult_expression = !import math; math.floor(1.5)
            ```

            ```pycon
            >>> import vectorbtpro as vbt

            >>> vbt.pprint(vbt.pdict.load("types.ini"))
            pdict(
                string='hello world',
                boolean=False,
                int=123,
                float=123.45,
                exp_float=1e-10,
                nan=np.nan,
                inf=np.inf,
                numpy=<numpy.ndarray object at 0x7fe1bf84f690 of shape (3,)>,
                pandas=<pandas.core.series.Series object at 0x7fe1c9a997f0 of shape (3,)>,
                expression=dict(
                    sub_dict2=dict(
                        some='value'
                    )
                ),
                mult_expression=1
            )
            ```

            * File `refs.ini`:

            ```ini
            [top]
            sr = &top.sr

            [top.sr @pandas.Series]
            data = [10756.12, 10876.76, 11764.33]
            index = &top.sr.index
            name = 'Open time'

            [top.sr.index @pandas.DatetimeIndex]
            data = ["2023-01-01", "2023-01-02", "2023-01-03"]
            ```

            ```pycon
            >>> vbt.pdict.load("refs.ini")["sr"]
            2023-01-01    10756.12
            2023-01-02    10876.76
            2023-01-03    11764.33
            Name: Open time, dtype: float64
            ```
        """
        import configparser
        from graphlib import TopologicalSorter

        if parser_kwargs is None:
            parser_kwargs = {}
        parser = configparser.RawConfigParser(**parser_kwargs)
        parser.optionxform = str

        try:
            parser.read_string(str_)
        except configparser.MissingSectionHeaderError as e:
            parser.read_string("[top]\n" + str_)

        def _get_path(k):
            if "@" in k:
                return k.split("@")[0].strip()
            return k

        dct = {}
        has_top_section = False
        for k in parser.sections():
            v = dict(parser.items(k))
            if _get_path(k) == "top":
                has_top_section = True
            elif not _get_path(k).startswith("top."):
                k = "top." + k
            new_v = {}
            for k2, v2 in v.items():
                if use_refs and v2.startswith("&") and not v2[1:].startswith("top."):
                    new_v[k2] = "&top." + v2[1:]
                else:
                    new_v[k2] = v2
            dct[k] = new_v
        if not has_top_section:
            dct = {"top": {"_": "_"}, **dct}

        def _get_class(k):
            if "@" in k:
                return k.split("@")[1].strip()
            return None

        class_map = {_get_path(k): _get_class(k) for k, v in dct.items()}
        dct = {_get_path(k): v for k, v in dct.items()}

        def _get_ref_node(ref):
            if ref in dct:
                ref_edges.add((k, (k, k2)))
                return ref
            ref_section = ".".join(ref.split(".")[:-1])
            ref_key = ref.split(".")[-1]
            if ref_section not in dct:
                raise ValueError(f"Referenced section '{ref_section}' not found")
            if ref_key not in dct[ref_section]:
                raise ValueError(f"Referenced object '{ref}' not found")
            return ref_section, ref_key

        # Parse config
        new_dct = dict()
        if code_context is None:
            code_context = {}
        code_context = {"np": np, "pd": pd, "vbt": vbt, **code_context}
        ref_edges = set()
        for k, v in dct.items():
            new_dct[k] = {}
            if len(v) == 1 and list(v.items())[0] == ("_", "_"):
                continue
            for k2, v2 in v.items():
                v2 = cls.decode_config_node(k2, v2, **kwargs)
                if isinstance(v2, str):
                    v2 = v2.strip()
                    if use_refs and v2.startswith("&"):
                        ref_node = _get_ref_node(v2[1:])
                        ref_edges.add((k, (k, k2)))
                        ref_edges.add(((k, k2), ref_node))
                    elif use_class_ids and v2.startswith("@"):
                        v2 = from_class_id(v2[1:])
                    elif run_code and v2.startswith("!"):
                        v2 = multiline_eval(v2.lstrip("!"), context=code_context)
                    else:
                        if (v2.startswith("'") and v2.endswith("'")) or (v2.startswith('"') and v2.endswith('"')):
                            v2 = v2[1:-1]
                        elif parse_literals:
                            if v2 == "np.nan":
                                v2 = np.nan
                            elif v2 == "np.inf":
                                v2 = np.inf
                            elif v2 == "-np.inf":
                                v2 = -np.inf
                            else:
                                try:
                                    v2 = ast.literal_eval(v2)
                                except Exception as e:
                                    try:
                                        v2 = float(v2)
                                    except Exception as e:
                                        pass
                new_dct[k][k2] = v2
        dct = new_dct

        # Build DAG
        graph = dict()
        keys = sorted(dct.keys())
        hierarchy = [keys[0]]
        for i in range(1, len(keys)):
            while True:
                if keys[i].startswith(hierarchy[-1] + "."):
                    if hierarchy[-1] not in graph:
                        graph[hierarchy[-1]] = set()
                    graph[hierarchy[-1]].add(keys[i])
                    hierarchy.append(keys[i])
                    break
                del hierarchy[-1]
        if use_refs and len(ref_edges) > 0:
            for k1, k2 in ref_edges:
                if k1 not in graph:
                    graph[k1] = set()
                graph[k1].add(k2)
        if len(graph) > 0:
            sorter = TopologicalSorter(graph)
            topo_order = list(sorter.static_order())

            # Resolve nodes
            resolved_nodes = dict()
            for k in topo_order:
                if isinstance(k, tuple):
                    v = dct[k[0]][k[1]]
                    if use_refs and isinstance(v, str) and v.startswith("&"):
                        ref_node = _get_ref_node(v[1:])
                        v = resolved_nodes[ref_node]
                else:
                    section_dct = dict(dct[k])
                    if k in graph:
                        for k2 in graph[k]:
                            if isinstance(k2, tuple):
                                section_dct[k2[1]] = resolved_nodes[k2]
                            else:
                                _k2 = k2[len(k) + 1:]
                                last_k = _k2.split(".")[-1]
                                d = section_dct
                                for s in _k2.split(".")[:-1]:
                                    if s not in d:
                                        d[s] = dict()
                                    d = d[s]
                                d[last_k] = resolved_nodes[k2]
                    if class_map.get(k, None) is not None and (pack_objects or k == "top"):
                        section_cls = class_map[k]
                        init_args = section_dct.pop("init_args~", ())
                        init_kwargs = section_dct.pop("init_kwargs~", {})
                        attr_dct = section_dct.pop("attr_dct~", {})
                        init_kwargs.update(section_dct)
                        rec_state = RecState(
                            init_args=init_args,
                            init_kwargs=init_kwargs,
                            attr_dct=attr_dct,
                        )
                        v = reconstruct(section_cls, rec_state)
                    else:
                        v = section_dct
                resolved_nodes[k] = v

            obj = resolved_nodes[topo_order[-1]]
        else:
            obj = dct["top"]
        if type(obj) is dict:
            obj = reconstruct(cls, RecState(init_kwargs=obj))
        if not isinstance(obj, cls):
            raise TypeError(f"Decoded object must be an instance of {cls}")
        return obj

    @classmethod
    def resolve_file_path(
        cls,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        for_load: bool = False,
    ) -> tp.Optional[Path]:
        """Resolve a file path.

        The following file formats are supported:

        * "pickle", "pkl", or None: Binary file
        * "config", "cfg", or "ini": Config
        """
        if path is None:
            path = cls.__name__
        path = Path(path)

        pickle_extensions = (".pickle", ".pkl")
        config_extensions = (".config", ".cfg", ".ini")
        if file_format is not None:
            if not file_format.startswith("."):
                file_format = "." + file_format
            if file_format.lower() in pickle_extensions:
                extensions = pickle_extensions
            elif file_format.lower() in config_extensions:
                extensions = config_extensions
            else:
                raise ValueError(f"Invalid file format '{file_format}'")
        else:
            extensions = pickle_extensions + config_extensions

        if for_load:
            if path.suffix == "":
                paths = []
                for k in extensions:
                    if path.with_suffix(k).exists():
                        paths.append(path.with_suffix(k))
                if len(paths) == 1:
                    return paths[0]
                if len(paths) > 1:
                    raise ValueError(f"Multiple paths founds: {paths}. Please provide an extension.")
                if len(paths) == 0:
                    return None
            else:
                if path.suffix not in extensions:
                    raise ValueError(f"Invalid file extension '{path.suffix}'")
                if path.exists():
                    return path
        else:
            if path.suffix == "":
                if file_format is not None:
                    return path.with_suffix(file_format)
                return path.with_suffix(extensions[0])
            else:
                if path.suffix not in extensions:
                    raise ValueError(f"Invalid file extension '{path.suffix}'")
                return path
        return None

    @classmethod
    def file_exists(
        cls,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
    ) -> bool:
        """Return whether a file already exists."""
        return cls.resolve_file_path(path=path, file_format=file_format, for_load=True) is not None

    def save(
        self,
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        mkdir_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        """Pickle/encode the instance and save to a file.

        Resolves the file path using `Pickleable.resolve_file_path`."""
        if mkdir_kwargs is None:
            mkdir_kwargs = {}

        path = self.resolve_file_path(path=path, file_format=file_format)
        if path.suffix in (".pickle", ".pkl"):
            bytes_ = self.dumps(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "wb") as f:
                f.write(bytes_)
        elif path.suffix in (".config", ".cfg", ".ini"):
            str_ = self.encode_config(**kwargs)
            check_mkdir(path.parent, **mkdir_kwargs)
            with open(path, "w") as f:
                f.write(str_)
        else:
            raise ValueError(f"Invalid file extension '{path.suffix}'")

    @classmethod
    def load(
        cls: tp.Type[PickleableT],
        path: tp.Optional[tp.PathLike] = None,
        file_format: tp.Optional[str] = None,
        **kwargs,
    ) -> PickleableT:
        """Unpickle/decode the instance from a file.

        Resolves the file path using `Pickleable.resolve_file_path`."""
        path = cls.resolve_file_path(path=path, file_format=file_format, for_load=True)
        if path.suffix in (".pickle", ".pkl"):
            with open(path, "rb") as f:
                bytes_ = f.read()
            return cls.loads(bytes_, **kwargs)
        elif path.suffix in (".config", ".cfg", ".ini"):
            with open(path, "r") as f:
                str_ = f.read()
            return cls.decode_config(str_, **kwargs)
        else:
            raise ValueError(f"Invalid file extension '{path.suffix}'")

    def __sizeof__(self) -> int:
        return len(self.dumps())

    def getsize(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get size of this object."""
        if readable:
            return humanize.naturalsize(self.__sizeof__(), **kwargs)
        return self.__sizeof__()

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        """Reconstruction state of the type `RecState`."""
        return None

    def __reduce__(self) -> tp.Union[str, tp.Tuple]:
        rec_state = self.rec_state
        if rec_state is None:
            return object.__reduce__(self)
        class_id = to_class_id(self)
        if class_id is None:
            cls = type(self)
        else:
            cls = class_id
        return reconstruct, (cls, rec_state)


pdictT = tp.TypeVar("pdictT", bound="pdict")


class pdict(Comparable, Pickleable, Prettified, dict):
    """Pickleable dict."""

    def load_update(self, path: tp.Optional[tp.PathLike] = None, clear: bool = False, **kwargs) -> None:
        """Load dumps from a file and update this instance in-place."""
        if clear:
            self.clear()
        self.update(self.load(path=path, **kwargs))

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        init_args = ()
        init_kwargs = dict(self)
        for k in list(init_kwargs):
            if not isinstance(k, str):
                if len(init_args) == 0:
                    init_args = (dict(),)
                init_args[0][k] = init_kwargs.pop(k)
        return RecState(init_args=init_args, init_kwargs=init_kwargs)

    def equals(self, other: tp.Any, check_types: bool = True) -> bool:
        """Check two objects for equality."""
        if check_types and type(self) != type(other):
            return False
        return is_deep_equal(dict(self), dict(other))

    def prettify(self, **kwargs) -> str:
        return prettify_dict(self, **kwargs)

    def __repr__(self):
        return type(self).__name__ + "(" + repr(dict(self)) + ")"
