"""Ordinary generated modules with fresh model/training global namespaces."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _fresh(name):
    path = Path(__file__).parent / "derived" / (name + ".py")
    spec = importlib.util.spec_from_file_location(__package__ + ".derived._instance_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_definitions():
    from .derived import graph
    return vars(graph)


def split_definitions():
    from .derived import splits
    return vars(splits)


def model_definitions(system):
    if str(system) not in {"14", "57", "118", "300"}:
        raise ValueError("Unsupported system")
    return vars(_fresh("network_" + str(system)))


def training_definition(config):
    module = _fresh("training")
    module.CONFIG = config
    return module.train_and_select


def evaluation_definition():
    return _fresh("training").test_selected


def optimized_definitions():
    from .derived import optimized
    return vars(optimized)


def sparse_definitions(optimized):
    module = _fresh("sparse")
    module.opt = SimpleNamespace(**optimized)
    return vars(module)
