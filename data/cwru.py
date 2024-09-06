from pathlib import Path
from typing import TypedDict
from enum import StrEnum, auto
import json
import numpy as np
from scipy.io import loadmat

from faultevent.data import DataLoader, Measurement
from faultevent.signal import Signal


class DataInfo(TypedDict):
    name: str
    id: str
    filename: str
    rpm: float
    hp: float


class CWRUDataLoader(DataLoader):

    def __init__(self, path):
        self.path = Path(path)#.parent
        with open("./data/cwru.json") as f:
            self.info = json.load(f)
    
    def signal_info(self, id) -> DataInfo | None:
        return next((d for d in self.info["data"] if d["id"] == id), None)

    def __getitem__(self, id) -> Measurement:
        info = self.signal_info(id)
        fn = info["filename"]
        mat = loadmat(self.path/fn)

        channel_base = f"X{info['id']}"
        x = np.squeeze(mat[channel_base+"_DE_time"])
        x -= np.mean(x)
        tx = np.arange(len(x))/self.info["fs"]
        vib = Signal(x, tx)

        t_end = len(x)/self.info["fs"]
        s = [0, t_end*info["rpm"]/60]
        ts = [0, t_end]
        pos = Signal(s, ts)

        return Measurement(vib, pos)



class Diagnostics(StrEnum):
    HEALTHY = "Healthy"
    ROLLER = "Roller"
    INNER = "Inner race"
    OUTER = "Outer race"



def fault(data_name: str) -> Diagnostics:
    if data_name[:6] == "Normal":
        return Diagnostics.HEALTHY
    elif data_name[:2] == "OR":
        return Diagnostics.OUTER
    elif data_name[0] == "B":
        return Diagnostics.ROLLER
    elif data_name[:2] == "IR":
        return Diagnostics.INNER
    return None