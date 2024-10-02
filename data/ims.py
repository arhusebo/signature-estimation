from pathlib import Path
from typing import TypedDict

import numpy as np

from faultevent.data import DataLoader, Measurement
from faultevent.signal import Signal


class SignalIdentifier(TypedDict):
    subpath: str
    name: str
    channel: int


fs = 20e3
rpm = 2000


class IMSDataLoader(DataLoader):

    def __init__(self, path):
        self.path = Path(path)

    def __getitem__(self, id_: SignalIdentifier) -> Measurement:
        path = self.path/id_["subpath"]/id_["name"]
        data = np.loadtxt(path)
        y = data[:,id_["channel"]]
        vib = Signal.from_uniform_samples(y, dx=1/fs)
        
        t_end = len(vib)/fs
        s = [0, t_end*rpm/60]
        ts = [0, t_end]
        pos = Signal(s, ts)

        return Measurement(vib, pos)


if __name__ == "__main__":
    from . import ims_path
    from faultevent.signal import ARModel

    dl = IMSDataLoader(ims_path)

    id_healthy: SignalIdentifier = {
        "subpath": "2nd_test",
        "name": "2004.02.12.10.32.39",
        "channel": 0
    }
    
    id_fault: SignalIdentifier = {
        "subpath": "2nd_test",
        "name": "2004.02.16.18.52.39",
        "channel": 0
    }

    model = ARModel.from_signal(dl[id_healthy].vib, p=50)


    signal = dl[id_fault].vib
    resid = model.residuals(signal)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(signal.x, signal.y)
    ax[1].plot(resid.x, resid.y)
    plt.show()