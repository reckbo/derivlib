A simple general python framework for encoding computations and their output in
a directed acyclic graph (DAG) of dependencies.  The main motivation for its
development is flexible output types, composable and queryable DAG's, and
auto-naming of outputs. Its use case is small to medium datasets; for example,
it currently serves as the foundation for a library of
neuroimaging pipelines extensively used in a research environment.

# Design goals

- output targets can be any part of an addressable store: a local filepath,
  database field, memory object, dropbox path, AWS path, list of filepaths, etc.
- transforms separate from the DAG: ability to test each transform individually, and re-use in different DAGs
- reusable and composible DAGs (e.g. ability to use sub-DAGs as inputs)
- ability to query the DAG for which outputs are complete and incomplete
- option for different schedulers (sequential, parallel, different conditional execution strategies)
  - currently only sequential implemented
- customizable method to specify how output data is determined to be up to date

# Non-goals:

- dynamically generated dependencies (dependencies generated from other dependencies during the running of a task)
  - this may be possible using an approach similiar to Luigi's

# Core Concepts

## 1. Resources

A resource represents a location, or set of locations, in an addressable store.
Often this is a local 
filepath, but it can be any location where data can be stored, such as memory or a database.

## 2. Transforms

Transforms are mappings from one or more resources to another resource.  They
encapsulate the logical transformation of data and are the computations that are
run by the scheduler.  They read the contents of their input resources, perform
some transformation, and write the results to an output resource.

## 3. Derivatives

Derivatives wrap transforms so that they can be composed into a DAG.  This means
that the output of one transform can be used as the input of another.  The child
nodes are source data, and the parent nodes are transforms who's outputs are
derived data.

# Example

``` python
from pathlib import Path

from derivlib import Transform, Source, Deriv
from derivlib.resources import LocalPath

class ConcatenateTxt(Transform):
  class Inputs:
    txt1: LocalPath
    txt2: LocalPath
  class Config:
    output_dir: Path | str
    
  def auto_output_id(self) -> str:
    return '_'.join(self.input_ids)
    
  def output(self) -> LocalPath:
    return LocalPath(self.output_id() + '.txt')
    
  def run(self):
    self.output().write_text(
      self.inputs.txt1.read_text() + self.inputs.txt2.read_text())

deriv = Deriv(
  ConcatenateTxt,
  inputs={'txt1': Source(LocalPath('/tmp/foo1.txt'), id='foo1'),
          'txt2': Source(LocalPath('/tmp/foo2.txt'), id='foo2')},
  config={'output_dir': '/tmp'}
)

Path('/tmp/foo1.txt').write_text('I am foo1')
Path('/tmp/foo2.txt').write_text(' and I am foo2')

deriv.show_outputs()  # prints tree of outputs, those in green are complete

# └─-ConcatenateTxt LocalPath(txt1_txt2.txt)
#   |--txt1: Source(LocalPath, id=foo1) LocalPath(/tmp/foo1.txt)
#   └─-txt2: Source(LocalPath, id=foo2) LocalPath(/tmp/foo2.txt)

deriv.make()  # Builds txt1_txt2.txt
deriv.make()  # Does nothing
deriv.output().read_text()

# I am foo1 and I am foo2
```


# TODO

- Add docstrings
- Add examples
- Implement a parallel scheduler similiar to `jug`.
