from typing import TypedDict, NotRequired, Literal
from enum import StrEnum
import tomllib
from functools import cache
from .uia import UiADataLoader
from .unsw import UNSWDataLoader
from .cwru import CWRUDataLoader


class Config(TypedDict):
    uia_path: NotRequired[str]
    unsw_path: NotRequired[str]
    cwru_path: NotRequired[str]


@cache
def load_config(path):
    global config
    with open(path, "rb") as fp:
        config = tomllib.load(fp)
    return config


def require_config(f, path="./config.toml"):
    config = load_config(path)
    def wrap(name: DataName):
        return f(config, name)
    return wrap


class DataName(StrEnum):
    UIA = "uia"
    UNSW = "unsw"
    CWRU = "cwru"


@require_config
def data_path(config: Config, name: DataName) -> str:
    match name:
        case DataName.UIA:
            dp = config["data"].get("uia", None)
        case DataName.UNSW:
            dp = config["data"].get("unsw", None)
        case DataName.CWRU:
            dp = config["data"].get("cwru", None)
        case _:
            raise ValueError("dataset name not recognized")
    if dp is None:
        raise ValueError("dataset path not configured")
    return dp


def dataloader(name: DataName):
    dp = data_path(name)
    match name:
        case DataName.UIA:
            return UiADataLoader(dp)
        case DataName.UNSW:
            return UNSWDataLoader(dp)
        case DataName.CWRU:
            return CWRUDataLoader(dp)
