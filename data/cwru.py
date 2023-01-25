from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat

from faultevent.data import DataLoader, Measurement
from faultevent.signal import Signal


class CWRUDataLoader(DataLoader):

    """Requires data files to be stored in the same directory along with
    an 'info.json' whose entries describe the information of each file.

    The key is the identifier used to load the signal.
    
    For example:
    
    {
        "97":
        {
            "name": "Normal_0",
            "id": "097",
            "filename": "97.mat",
            "rpm": 1797,
            "hp": 0
        },
        ...
    }

    """


    def __init__(self, path):
        self.path = Path(path)#.parent
        with open(self.path/"info.json") as f:
            self.info = json.load(f)
        self.info["fs"] = 48.e3

    def _matfile(self, id):
        fn = self.info[str(id)]["filename"]
        mat = loadmat(self.path/fn)
        return mat

    def __getitem__(self, id) -> Measurement:
        mat = self._matfile(id)

        channel_base = f"X{self.info[str(id)]['id']}"
        x = np.squeeze(mat[channel_base+"_DE_time"])
        x -= np.mean(x)
        tx = np.arange(len(x))/self.info["fs"]
        vib = Signal(x, tx)

        t_end = len(x)/self.info["fs"]
        s = [0, t_end*self.info[str(id)]["rpm"]/60]
        ts = [0, t_end]
        pos = Signal(s, ts)

        return Measurement(vib, pos)
