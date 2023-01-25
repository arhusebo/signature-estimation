from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from faultevent.data import DataLoader, Measurement
from faultevent.signal import Signal


class UNSWDataLoader(DataLoader):
    def __init__(self, path = Path(".")):
        self.path = Path(path)

    def __getitem__(self, subpath: str) -> Measurement:
        mat = loadmat(self.path/subpath)

        fs = np.squeeze(mat["Fs"])
        x = np.squeeze(mat["accV"])
        x -= np.mean(x)
        vib = Signal.from_uniform_samples(x, 1/fs)

        s = np.squeeze(mat["enc1"])
        peaks, _ = find_peaks(s, 2.5)
        ts = peaks/fs
        s = np.arange(len(peaks))/1024
        pos = Signal(s, ts)

        return Measurement(vib, pos)
