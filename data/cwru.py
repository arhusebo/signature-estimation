from pathlib import Path
from typing import TypedDict
from enum import StrEnum
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
        vib = Signal.from_uniform_samples(x, 1/self.info["fs"])

        t_end = len(x)/self.info["fs"]
        s = [0, t_end*info["rpm"]/60]
        ts = [0, t_end]
        pos = Signal(s, ts, uniform_samples=True)

        return Measurement(vib, pos)



class Diagnostics(StrEnum):
    HEALTHY = "Healthy"
    ROLLER = "Roller"
    INNER = "Inner race"
    OUTER = "Outer race"



def fault(data_name: str) -> Diagnostics | int:
    if data_name[:6] == "Normal":
        mm = 0
        return Diagnostics.HEALTHY, mm
    elif data_name[:2] == "OR":
        mm = int(data_name[2:5])
        return Diagnostics.OUTER, mm
    elif data_name[0] == "B":
        mm = int(data_name[1:4])
        return Diagnostics.ROLLER, mm
    elif data_name[:2] == "IR":
        mm = int(data_name[2:5])
        return Diagnostics.INNER, mm
    return None