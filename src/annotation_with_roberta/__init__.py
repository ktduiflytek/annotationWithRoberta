"""Utilities for training and running the multilingual slot annotation model."""

from importlib import import_module

from . import data

__all__ = ["data"]


def __getattr__(name):  # pragma: no cover - dynamic re-export
    if name == "training":
        module = import_module("annotation_with_roberta.training")
        globals()[name] = module
        return module
    if name == "inference":
        module = import_module("annotation_with_roberta.inference")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
