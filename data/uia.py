from pathlib import Path
import h5py

from faultevent.data import DataLoader, Measurement
from faultevent.signal import Signal

import numpy as np

class UiADataLoader(DataLoader):
    def __init__(self, path = Path(".")):
        self.path = Path(path)

    def __getitem__(self, subpath: str) -> Measurement:
        path = self.path/subpath
        f = h5py.File(path, "r")

        fs = f["vib"].attrs["Fs"]
        signal = f["vib"]*f["vib"].attrs["scale"]
        x = np.asarray(signal)
        x -= np.mean(x)
        vib = Signal.from_uniform_samples(x, 1/fs)

        s = np.array(f["s"])*f["s"].attrs["scale"]
        pos = Signal.from_uniform_samples(s, 1/f["s"].attrs["Fs"])

        return Measurement(vib, pos)
