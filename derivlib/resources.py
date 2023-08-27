from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from typing import Any

from .base import Resource


class Resources(Resource):
    @classmethod
    def from_list(cls, resources):
        return cls({str(k): v for k, v in enumerate(resources)})

    def __init__(self, resources: dict | pd.Series):
        if isinstance(resources, pd.Series):
            resources = resources.to_dict()
        self.resources = resources

    def exists(self) -> bool:
        return all(res.exists() for res in self.resources.values())

    def status(self) -> dict:
        return {k: res.exists() for k, res in self.resources.items()}


class Value(Resource):
    def __init__(self, val: Any):
        self.val = val

    def __repr__(self):
        return f"{self.__class__.__name__}({self.val})"

    def read(self) -> Any:
        return self.val

    def write(self, new_val) -> None:
        self.val = new_val

    def exists(self):
        return self.val is not None


class LocalPath(Resource):
    def __init__(self, *path_segments):
        self._path = Path(*path_segments)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return False
        return other.path == self.path

    @property
    def path(self) -> Path:
        return self._path

    def with_suffix(self, *args) -> LocalPath:
        return self.__class__(self.path.with_suffix(*args))

    def exists(self) -> bool:
        return self.path.exists()

    def unlink(self):
        self.path.unlink()

    def write_text(self, data, encoding=None):
        self.path.write_text(data, encoding=encoding)

    def read_text(self, encoding=None) -> str:
        return self.path.read_text(encoding=encoding)


class LocalPaths(Resource):
    def __init__(self, paths: dict, repr_path=None):
        self._paths = {k: Path(v) for k, v in paths.items()}
        self._repr_path = repr_path or self.paths

    def __repr__(self):
        return f"{self.__class__.__name__}({self._repr_path})"

    def __eq__(self, other):
        if other.__class__ != self.__class__:
            return False
        return other.paths == self.paths

    @property
    def paths(self):
        return self._paths

    def exists(self):
        return all(p.exists() for p in self.paths.values())

    def status(self):
        return {p: p.exists() for p in self.paths.values()}


class ReadableResource(Resource):
    @abstractmethod
    def read(self):
        pass
