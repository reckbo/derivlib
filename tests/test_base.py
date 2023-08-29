from pathlib import Path
import time
import pytest

from derivlib.base import Resource, Source, Deriv, Transform, Wrapper
from derivlib.resources import Value


class Add(Transform):
    class Inputs:
        op1: Resource
        op2: Resource

    class Params:
        funky: bool = False

    class Config:
        output_dir: Path = Path("/outdir")

    def auto_output_id(self):
        return "_".join(["add", self.input_ids["op1"], self.input_ids["op2"]])

    def output(self):
        return Path(self.config.output_dir, self.output_id() + ".txt")

    def run(self):
        self.output()


@pytest.fixture
def val2():
    return Source(Value(2), id="val2")


@pytest.fixture
def val5():
    return Source(Value(5), id="val5")


@pytest.fixture
def add(val2, val5):
    return Deriv(Add, inputs={"op1": val2, "op2": val5})


@pytest.fixture
def neg2():
    return Source(Value(-2), id="neg2")


@pytest.fixture
def neg5():
    return Source(Value(-5), id="neg5")


@pytest.fixture
def add_neg(neg2, neg5):
    return Deriv(Add, inputs={"op1": neg2, "op2": neg5})


class TestOutputs:
    def test_auto_output_id(self, add):
        assert add.output_id() == "add_val2_val5"
        assert add.output() == Path("/outdir/add_val2_val5.txt")

    def test_overriding_output_id(self, val2, val5):
        deriv = Deriv(Add, inputs={"op1": val2, "op2": val5}, output_id="pants")
        assert deriv.output_id() == "pants"
        assert deriv.output() == Path("/outdir/pants.txt")


class TestShowOutputs:
    def test_external_repr_output(self, val2):
        assert val2.repr_output() == "Value(2)"

    def test_deriv_repr_output(self, add):
        assert add.repr_output() == "/outdir/add_val2_val5.txt"

    def test_wrapper_repr_output(self, add):
        wrapper = Wrapper.from_kwargs(add=add)
        assert wrapper.repr_output() == ""

    def test_external_show_outputs(self, capsys, val2):
        val2.show_outputs()
        captured = capsys.readouterr()
        assert len(captured.out) == 45

    def test_deriv_show_outputs(self, capsys, add):
        add.show_outputs()
        captured = capsys.readouterr()
        assert len(captured.out) == 160


def test_toposort(val2, val5, add):
    assert add.toposort() == [val2, val5, add]


def test_iter(add):
    for i, n in enumerate(add):
        assert n is add.toposort()[i]


class Test_Wrapper:
    def test_create_basic(self, val2, val5, add):
        wrapper = Wrapper.from_kwargs(add=add)
        assert wrapper.toposort() == [val2, val5, add, wrapper]

    def test_create_from_list(self, val2, val5, add):
        wrapper = Wrapper.from_list([add])
        assert wrapper.toposort() == [val2, val5, add, wrapper]

    def test_get_using_getitem(self, val2, val5, add):
        wrapper = Wrapper.from_kwargs(two=val2, five=val5, add=add)
        assert wrapper["two"] is val2
        assert wrapper["five"] is val5
        assert wrapper["add"] is add


class TestSpeed:
    def test_show_outputs_speed(self, val2):
        derivs = [val2]
        for _ in range(1, 100):
            deriv = Deriv(
                Add,
                inputs={"op1": derivs[-1], "op2": val2},
                params={"funky": True},
                config={"output_dir": Path("/output")},
            )
            derivs.append(deriv)
        t0 = time.time()
        derivs[-1].o
        t1 = time.time()
        delta = t1 - t0
        assert delta < 0.5

    def test_show_configs_speed(self, val2):
        derivs = [val2]
        for _ in range(1, 100):
            deriv = Deriv(
                Add,
                inputs={"op1": derivs[-1], "op2": val2},
                params={"funky": True},
                config={"output_dir": Path("/output")},
            )
            derivs.append(deriv)
        t0 = time.time()
        derivs[-1].show_configs()
        t1 = time.time()
        delta = t1 - t0
        assert delta < 0.5
