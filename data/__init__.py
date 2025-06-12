from typing import TypedDict, NotRequired
from enum import StrEnum
from pathlib import Path
from .uia import UiADataLoader
from .unsw import UNSWDataLoader
from .cwru import CWRUDataLoader

from config import load_config

class DatasetConfig(TypedDict):
    uia_path: NotRequired[str]
    unsw_path: NotRequired[str]
    cwru_path: NotRequired[str]


def require_config(f, path="./config.toml"):
    config = load_config(path)
    def wrap(name: DataName):
        return f(config["data"], name)
    return wrap


class DataName(StrEnum):
    UIA = "uia"
    UNSW = "unsw"
    CWRU = "cwru"


@require_config
def data_path(config: DatasetConfig, name: DataName) -> Path:
    match name:
        case DataName.UIA:
            dp = config.get(DataName.UIA, None)
        case DataName.UNSW:
            dp = config.get(DataName.UNSW, None)
        case DataName.CWRU:
            dp = config.get(DataName.CWRU, None)
        case _:
            raise ValueError("dataset name not recognized")
    if dp is None:
        raise ValueError("dataset path not configured")
    return Path(dp)


def dataloader(name: DataName):
    dp = data_path(name)
    match name:
        case DataName.UIA:
            return UiADataLoader(dp)
        case DataName.UNSW:
            return UNSWDataLoader(dp)
        case DataName.CWRU:
            return CWRUDataLoader(dp)
