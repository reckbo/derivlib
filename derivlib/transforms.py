from pathlib import Path
import logging
import subprocess
from typing import Optional, Callable, List

import pandas as pd

from neurolib.resources import Csv, Nifti, LocalPath
from .base import Resource, Transform

logger = logging.getLogger(__name__)


class ProjectResource(Transform):
    """Creates one resource from another, but doesn't perform any logic or have any side effects.

    Raises
    ------
    RuntimeError
    """

    class Inputs:
        resource: Resource

    class Params:
        id: str

    class Config:
        selector: Callable[[Resource], Resource]

    @property
    def name(self):
        name = self.params.id
        if name:
            return f"ProjectResource [{self.params.id}]"
        return super().name

    def output(self) -> Resource:
        return self.config.selector(self.inputs.resource)

    def run(self):
        if self.output().exists():
            logger.info("%s exists", self.output())
        else:
            raise RuntimeError(
                f"Selected resource does not exist: {self.output()} (parent resource: {self.inputs.resource})"
            )

    def auto_output_id(self) -> str:
        return self.input_ids["resource"] + "/" + self.params.id


class SymlinkNifti(Transform):
    class Inputs:
        nifti: Nifti

    class Params:
        symlink_name: str

    class Config:
        output_dir: Optional[Path] = None
        relative: bool = False

    def auto_output_id(self) -> str:
        return self.params.symlink_name

    def _output_dir(self):
        if self.config.output_dir:
            return Path(self.config.output_dir)
        return Path(self.inputs.pet.paths["nifti"]).parent

    def output(self) -> Nifti:
        input_suffixes = self.inputs.nifti.path.suffixes
        if input_suffixes == [".nii"]:
            input_suffix = ".nii"
        elif input_suffixes == [".nii", ".gz"]:
            input_suffix = ".nii.gz"
        else:
            raise ValueError(
                "Expect input nifti to have extension .nii or .nii.gz, but got: {self.inputs.nifti.path}"
            )
        return Nifti(self._output_dir() / (self.output_id() + input_suffix))

    def run(self):
        target_nifti = self.inputs.nifti.path.resolve()
        symlink_nifti = self.output().path.resolve()
        if self.config.relative:
            target_nifti = target_nifti.relative_to(symlink_nifti.parent)
        try:
            symlink_nifti.unlink()
        except FileNotFoundError:
            pass
        logger.info(
            "%s: Make symlink from %s to %s", self.name, symlink_nifti, target_nifti
        )
        if not symlink_nifti.parent.exists():
            symlink_nifti.parent.mkdir(parents=True)
        symlink_nifti.symlink_to(target_nifti)
        logger.info("%s: Made %s", self.name, self.output())


class UnTar(Transform):
    class Inputs:
        file: LocalPath

    class Config:
        output_dir: Optional[Path] = None

    def auto_output_id(self) -> str:
        suffixes = self.inputs.file.path.suffixes
        if suffixes == [".tar"]:
            raise NotImplementedError("Need to implement for .tar")
        if suffixes != [".tar", ".gz"]:
            raise ValueError(
                f"Expected .tar.gz suffix, but got: {self.inputs.file.path}"
            )
        return self.inputs.file.path.with_suffix("").with_suffix("").stem

    def _output_dir(self):
        if self.config.output_dir:
            return Path(self.config.output_dir)
        return Path(self.inputs.file.path).parent

    def output(self) -> LocalPath:
        return LocalPath(self._output_dir(), self.output_id())

    def _command(self) -> List[str]:
        return [
            "tar",
            "zxvf",
            str(self.inputs.file.path),
            "-C",
            str(self.output().path),
        ]

    def run(self):
        logger.info("Extract %s", self.inputs.file.path)
        self.output().path.mkdir(parents=True)
        subprocess.run(self._command(), check=True)
        logger.info("Made: %s", self.output().path)


class TransformCsv(Transform):
    class Inputs:
        csv: Csv

    class Params:
        id: str

    class Config:
        func: Callable
        output_dir: Path

    def auto_output_id(self):
        return self.input_ids["csv"] + "_" + self.params.id

    def output(self) -> Csv:
        return Csv(
            Path(self.config.output_dir, self.output_id() + ".csv"),
            index_col=self.inputs.csv.index_col,
        )

    def _make_df(self) -> pd.DataFrame:
        return self.inputs.csv.read().pipe(self.config.func)

    def run(self):
        self.output().write(self._make_df())


class JoinCsvs(Transform):
    class Inputs:
        left: Csv
        right: Csv

    class Config:
        output_dir: Path

    def output(self) -> Csv:
        left = self.inputs.left
        right = self.inputs.right
        if left.index_col != right.index_col:
            raise ValueError(
                "Input csv's have differing index columns:\n",
                f"left csv: {left.index_col}\nright csv: {right.index_col}",
            )
        return Csv(
            Path(self.config.output_dir, self.output_id() + ".csv"),
            index_col=left.index_col,
        )

    def _make_df(self):
        return self.inputs.left.read().join(self.inputs.right.read())

    def run(self):
        self.output().write(self._make_df())
