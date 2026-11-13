"""Headset-agnostic EEG motor imagery -> motion command toolkit."""
from .config import load_config
from .data import Recording

__all__ = ["load_config", "Recording"]
