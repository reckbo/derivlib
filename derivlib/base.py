from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import logging
import dataclasses
from dataclasses import dataclass
from pprint import pformat


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
    node, get_children, indent="", last=True, repr_node_fn=repr, depth=0, max_depth=100
):
    result = "\n" + indent
    if last:
        result += "└─-"
        indent += "   "
    else:
        result += "|--"
        indent += "|  "
    result += repr_node_fn(node)
    if depth >= max_depth:
        return result
    children = get_children(node)
    for idx, child in enumerate(children):
        result += repr_tree(
            node=child,
            get_children=get_children,
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


def concat(lst):
    return [item for sublist in lst for item in sublist]


def _toposort(node):
    return concat(_toposort(op) for op in node.deps.values()) + [node]


def _repr_dict(dct):
    if not dct:
        return ""
    return "(" + ", ".join([f"{k}={v}" for k, v in dct.items()]) + ")"


class Resource(ABC):
    @abstractmethod
    def exists(self) -> bool:
        pass

    def status(self) -> dict:
        return {"exists": self.exists()}


class Transform(ABC):
    def __init__(
        self, inputs, params=None, config=None, input_ids=None, output_id=None
    ):
        self._input_kwargs = inputs
        self._param_kwargs = params or {}
        self._config_kwargs = config or {}
        self._output_id = output_id
        self.input_ids = input_ids or {}
        self._inputs = self._bind_inputs(self._input_kwargs)
        self._params = self._bind_params(self._param_kwargs)
        self._config = self._bind_config(self._config_kwargs)

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
    def get_config_dataclass(cls):
        return dataclass(cls.Config)

    @classmethod
    def get_params_dataclass(cls):
        return dataclass(cls.Params)

    @staticmethod
    def make_model(model_cls, kwargs):
        try:
            return model_cls(**kwargs)
        except TypeError as err:
            raise TypeError(f"{model_cls}: {err}") from err

    def auto_output_id(self):
        raise NotImplementedError(
            f'Implement "auto_output_id" if you want this transform ({self.__class__}) to auto-generate its output location'
        )

    def output_id(self) -> str:
        if self._output_id:
            return self._output_id
        return self.auto_output_id()

    @abstractmethod
    def output(self) -> Resource:
        pass

    def _bind_inputs(self, kwargs):
        return self.make_model(self.get_inputs_dataclass(), kwargs)

    def _bind_params(self, kwargs):
        return self.make_model(self.get_params_dataclass(), kwargs)

    def _bind_config(self, kwargs):
        return self.make_model(self.get_config_dataclass(), kwargs)

    @property
    def inputs(self):
        return self._inputs

    @property
    def params(self):
        return self._params

    @property
    def config(self):
        return self._config

    def params_dict(self):
        return dataclasses.asdict(self.params)

    def config_dict(self):
        return dataclasses.asdict(self.config)

    def inputs_dict(self):
        return dataclasses.asdict(self.inputs)

    def with_inputs(self, **kwargs):
        return self.__class__(
            inputs=dict(self._input_kwargs, **kwargs),
            params=self._param_kwargs,
            config=self._config_kwargs,
            input_ids=self.input_ids,
            output_id=self._output_id,
        )

    def with_params(self, **kwargs):
        return self.__class__(
            inputs=self._input_kwargs,
            params=dict(self._param_kwargs, **kwargs),
            config=self._config_kwargs,
            input_ids=self.input_ids,
            output_id=self._output_id,
        )

    def with_config(self, **kwargs):
        return self.__class__(
            inputs=self._input_kwargs,
            params=self._param_kwargs,
            config=dict(self._config_kwargs, **kwargs),
            input_ids=self.input_ids,
            output_id=self._output_id,
        )

    def with_input_ids(self, **kwargs):
        return self.__class__(
            inputs=self._input_kwargs,
            params=self._param_kwargs,
            config=self._config_kwargs,
            input_ids=dict(self.input_ids, **kwargs),
            output_id=self._output_id,
        )

    def with_output_id(self, output_id):
        return self.__class__(
            inputs=self._input_kwargs,
            params=self._param_kwargs,
            config=self._config_kwargs,
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
                get_children=lambda x: x.deps.values(),
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
        self._show(
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
        result = _toposort(self)
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


class Deriv(Node):
    def __init__(
        self,
        transform_cls,
        *,
        inputs=None,
        params=None,
        config=None,
        output_id=None,
        tag=None,
    ):
        self._transform_cls = transform_cls
        self._input_kwargs = inputs or {}
        self._param_kwargs = params or {}
        self._config_kwargs = config or {}
        self._given_output_id = output_id
        self.tag = tag
        self.transform = self._create_transform()

    def _create_transform(self):
        return self._transform_cls(
            inputs={k: n.output() for k, n in self._input_kwargs.items()},
            params=self._param_kwargs,
            config=self._config_kwargs,
            input_ids={k: n.output_id() for k, n in self._input_kwargs.items()},
            output_id=self._given_output_id,
        )

    def __repr__(self):
        params = _repr_dict(self.transform.params_dict())
        return f"{self.transform.name}{params}"

    @property
    def name(self):
        return self.transform.name

    @property
    def deps(self):
        return self._input_kwargs

    @property
    def params(self):
        return self.transform.params_dict()

    @property
    def config(self):
        return self.transform.config_dict()

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