from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import dataclasses
from dataclasses import dataclass
import itertools
import logging
from pprint import pformat
from typing import Any, List, Callable


logger = logging.getLogger(__name__)


class StringColor:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    END = "\033[0m"

    @staticmethod
    def green(astring):
        return StringColor.GREEN + astring + StringColor.END

    @staticmethod
    def blue(astring):
        return StringColor.BLUE + astring + StringColor.END


def repr_tree(
    node: Any,
    get_children: Callable,
    prefix: str = "",
    indent: str = "",
    last: bool = True,
    repr_node_fn: Callable = repr,
    depth: int = 0,
    max_depth: int = 100,
):
    """Create string representation of a DAG.

    Parameters
    ----------
    node : Any
        node of a DAG
    get_children : Callable
        function that returns dict of children of a node
    prefix : str
        prefix string of node representation
    indent : string
        indentation of node
    last : bool
        is it the last node?
    repr_node_fn : Callable
        function that returns string representation of a node
    depth : int
    max_depth : int
    """
    result = "\n" + indent
    if last:
        result += "└─-"
        indent += "   "
    else:
        result += "|--"
        indent += "|  "
    result += prefix + repr_node_fn(node)
    if depth >= max_depth:
        return result
    children = get_children(node)
    for idx, (name, child) in enumerate(children.items()):
        result += repr_tree(
            node=child,
            get_children=get_children,
            prefix=name + ": ",
            indent=indent,
            last=(idx + 1) == len(children),
            repr_node_fn=repr_node_fn,
            depth=depth + 1,
            max_depth=max_depth,
        )
    return result


def filter_unique(xs, key_func=id):
    seen = set()
    result = []
    for x in xs:
        if key_func(x) not in seen:
            result.append(x)
        seen.add(key_func(x))
    return result


def concat(lsts) -> List:
    return list(itertools.chain(*lsts))


def toposort(node: Node):
    return concat(toposort(op) for op in node.deps.values()) + [node]


def _repr_dict(dct):
    if not dct:
        return ""
    return "(" + ", ".join([f"{k}={v}" for k, v in dct.items()]) + ")"


class Resource(ABC):
    """
    Abstract class representing a "resource".

    A resource represents a location, or set of locations, in an addressable
    store.  Often this is a local  filepath, but it can be any location where
    data can be stored, such as memory or a database.
    """

    @abstractmethod
    def exists(self) -> bool:
        """Is the resource location non-empty?"""
        pass

    def status(self) -> dict:
        """Returns an overview."""
        return {"exists": self.exists()}


def _instantiate_dataclass(dataclass_cls, kwargs):
    try:
        return dataclass_cls(**kwargs)
    except TypeError as err:
        raise TypeError(f"{dataclass_cls}: {err}") from err


def _bind_kwargs(dataclass_cls, kwargs: dict) -> dict:
    return dataclasses.asdict(_instantiate_dataclass(dataclass_cls, kwargs))


class Transform(ABC):
    """
    Abstract class for a "transform".

    Transforms are mappings from one or more resources to another resource.
    They encapsulate the logical transformation of data and are the computations
    that are run by the scheduler.  They read the contents of their input
    resources, perform some transformation, and write the results to an output
    resource.

    Parameters
    ----------
    inputs : dict
        input resources
    params : dict
        parameters that modify the computation
    config : dict
        values that control how the computation is run, but does not have an
        effect on the output
    input_ids : dict
        string IDs for each of the inputs
    output_id : str
        string ID for the output data

    Attributes
    ----------
    inputs
        instantiated Inputs class
    params
        instatiated Params class
    config
        instantiated Config class
    input_ids
        dict of ID's for the input data that may be used to auto-generate an output ID
    given_output_id
        user supplied output ID that overrides the auto-generated output ID

    Raises
    ------
    NotImplementedError
    TypeError
    """

    inputs: Inputs
    params: Params
    config: Config
    input_ids: dict | None
    given_output_id: str | None

    def __init__(
        self,
        inputs: dict,
        params: dict | None = None,
        config: dict | None = None,
        input_ids: dict | None = None,
        output_id: str | None = None,
    ):

        self.input_kwargs = inputs
        self.param_kwargs = params or {}
        self.config_kwargs = config or {}
        self.given_output_id = output_id
        self.input_ids = input_ids
        self.inputs = _instantiate_dataclass(self.get_inputs_dataclass(), inputs)
        self.params = _instantiate_dataclass(
            self.get_params_dataclass(), self.param_kwargs
        )
        self.config = _instantiate_dataclass(
            self.get_config_dataclass(), self.config_kwargs
        )

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def qualname(self):
        return self.__class__.__module__ + "." + self.__class__.__name__

    class Inputs:
        pass

    class Params:
        pass

    class Config:
        pass

    @classmethod
    def get_inputs_dataclass(cls):
        return dataclass(cls.Inputs)

    @classmethod
    def get_params_dataclass(cls):
        return dataclass(cls.Params)

    @classmethod
    def get_config_dataclass(cls):
        return dataclass(cls.Config)

    def auto_output_id(self):
        raise NotImplementedError(
            f'Implement "auto_output_id" if you want this transform ({self.__class__}) to auto-generate its output location'
        )

    def output_id(self) -> str:
        if self.given_output_id:
            return self.given_output_id
        return self.auto_output_id()

    @abstractmethod
    def output(self) -> Resource:
        pass

    @property
    def inputs_dict(self):
        return dataclasses.asdict(self.inputs)

    @property
    def params_dict(self):
        return dataclasses.asdict(self.params)

    @property
    def config_dict(self):
        return dataclasses.asdict(self.config)

    def with_inputs(self, **kwargs):
        return self.__class__(
            inputs=dict(self.input_kwargs, **kwargs),
            params=self.param_kwargs,
            config=self.config_kwargs,
            input_ids=self.input_ids,
            output_id=self._output_id,
        )

    def with_params(self, **kwargs):
        return self.__class__(
            inputs=self.input_kwargs,
            params=dict(self.param_kwargs, **kwargs),
            config=self._config_kwargs,
            input_ids=self.input_ids,
            output_id=self._given_output_id,
        )

    def with_config(self, **kwargs):
        return self.__class__(
            inputs=self.input_kwargs,
            params=self.param_kwargs,
            config=dict(self.config_kwargs, **kwargs),
            input_ids=self.input_ids,
            output_id=self._output_id,
        )

    def with_input_ids(self, **kwargs):
        return self.__class__(
            inputs=self.input_kwargs,
            params=self.param_kwargs,
            config=self.config_kwargs,
            input_ids=dict(self.input_ids, **kwargs),
            output_id=self._output_id,
        )

    def with_output_id(self, output_id):
        return self.__class__(
            inputs=self.input_kwargs,
            params=self.param_kwargs,
            config=self.config_kwargs,
            input_ids=self.input_ids,
            output_id=output_id,
        )

    @abstractmethod
    def run(self):
        pass


class Node(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def params(self):
        return {}

    @property
    def config(self):
        return {}

    @abstractmethod
    def output_id(self):
        pass

    @abstractmethod
    def output(self):
        pass

    @property
    @abstractmethod
    def deps(self):
        pass

    @abstractmethod
    def uptodate(self):
        pass

    @abstractmethod
    def make(self):
        pass

    def repr_node(self):
        return repr(self)

    def _show_tree(self, depth=100, repr_node_fn=repr):
        print(
            repr_tree(
                self,
                get_children=lambda x: x.deps,
                repr_node_fn=repr_node_fn,
                max_depth=depth,
            )
        )

    def show(self, depth=100, color=True):
        if color:
            repr_node = lambda n: (
                StringColor.green if n.uptodate() else StringColor.blue
            )(n.repr_node())
        else:
            repr_node = repr
        self._show_tree(depth, repr_node_fn=repr_node)

    def repr_config(self):
        return f"{self.config}"

    def repr_output(self):
        return f"{self.output()}"

    def show_configs(self, depth=100):
        self._show_tree(depth, lambda x: f"{x} {x.repr_config()}")

    def show_outputs(self, depth=100):
        self._show_tree(
            depth,
            lambda n: (StringColor.green if n.uptodate() else StringColor.blue)(
                f"{n.repr_node()} {n.repr_output()}"
            ),
        )

    def show_output_ids(self, depth=100):
        self._show_tree(
            depth,
            lambda x: (StringColor.green if x.uptodate() else StringColor.blue)(
                f"{x.repr_node()}  {x.output_id()}"
            ),
        )

    @property
    def o(self):
        return self.show_outputs()

    @property
    def st(self):
        return self.show(depth=100, color=True)

    def toposort(self, unique=True):
        result = toposort(self)
        if unique:
            return filter_unique(result)
        return result

    def __iter__(self):
        yield from self.toposort()


class Source(Node):
    """Wraps a given resource."""

    def __init__(self, resource, id=None, tag=None):
        self.resource = resource
        self.id = id
        self.tag = tag

    def __repr__(self):
        return f"Source({self.resource.__class__.__name__}, id={self.id})"

    @property
    def name(self):
        return self.resource.__class__.__name__

    @property
    def qualname(self):
        return self.__class__.__module__ + "." + self.name

    @property
    def deps(self):
        return {}

    @property
    def params(self):
        return {"id": self.id}

    def output(self):
        return self.resource

    def output_id(self):
        return self.id

    def uptodate(self):
        return self.resource.exists()

    def status(self):
        return self.resource.status()

    def make(self):
        if self.uptodate():
            return
        raise Exception(f"Missing external data: {self.resource}")


def _create_transform(
    transform_cls, inputs: dict, params: dict, config: dict, output_id=None
):
    return transform_cls(
        inputs={k: n.output() for k, n in inputs.items()},
        params=params,
        config=config,
        input_ids={k: n.output_id() for k, n in inputs.items()},
        output_id=output_id,
    )


class Deriv(Node):
    def __init__(
        self,
        transform_cls,
        *,
        inputs=None,
        params=None,
        config=None,
        output_id=None,
    ):
        self._deps = inputs
        self.transform = _create_transform(
            transform_cls, inputs, params, config, output_id
        )

    def __repr__(self):
        params = _repr_dict(self.transform.params_dict)
        return f"{self.transform.name}{params}"

    @property
    def name(self):
        return self.transform.name

    @property
    def deps(self):
        return self._deps

    @property
    def params(self):
        return self.transform.params_dict

    @property
    def config(self):
        return self.transform.config_dict

    def output_id(self):
        return self.transform.output_id()

    def output(self):
        return self.transform.output()

    def uptodate(self):
        return self.transform.output().exists()

    # TODO move this simple scheduler into its own function in preparation for
    # adding a concurrent version as an option
    def make(self):
        logger.info("%s: Build", self)
        if self.uptodate():
            logger.info("%s: Already up to date.", self)
            return
        logger.info("%s: Build deps", self)
        for dep in self.deps.values():
            dep.make()
        logger.info("%s: Run transform: %s", self, self.transform)
        self.transform.run()
        logger.info("%s: Done.", self)


class Wrapper(Node, Mapping):
    @classmethod
    def from_list(cls, nodes, description=None, tag=None):
        return cls(
            {str(k): v for k, v in enumerate(nodes)}, description=description, tag=tag
        )

    @classmethod
    def from_kwargs(cls, *, description=None, tag=None, **deps):
        return cls(deps, description=description, tag=tag)

    def __init__(self, deps: dict | None = None, description=None, tag=None, id=None):
        self._deps = deps or {}
        self.tag = tag
        self.description = description or ""
        self._id = id

    # TODO fix indenting
    def __repr__(self):
        return f"{self.__class__.__name__}(\n{pformat(self.deps, sort_dicts=False)})"

    def __getitem__(self, key):
        return self.deps[key]

    def __setitem__(self, key, dep):
        self.deps[key] = dep

    def __iter__(self):
        return iter(self.deps)

    def __len__(self):
        return len(self.deps)

    def items(self):
        return self.deps.items()

    def append(self, deps):
        for k, v in deps.items():
            assert k not in self.deps
            self.deps[k] = v

    def repr_node(self):
        return f"Wrapper({self.description})"

    @property
    def name(self):
        return "Wrapper"

    @property
    def deps(self):
        return self._deps

    def output(self):
        return {k: n.output() for k, n in self.deps.items()}

    def repr_output(self):
        return ""

    def output_id(self):
        return self._id

    def uptodate(self):
        return all(n.uptodate() for n in self.deps.values())

    def make(self):
        logger.info("%s: Build", self)
        if self.uptodate():
            logger.info("%s: Already up to date.", self)
            return
        logger.info("%s: Build deps", self)
        for dep in self.deps.values():
            dep.make()

    def status(self) -> dict:
        return {k: deriv.uptodate() for k, deriv in self.items()}
