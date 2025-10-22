from typing import Sequence, TypedDict
from collections.abc import Iterator
from functools import partial
import itertools
import pathlib
import numpy as np
import torch

import data
from config import load_config

from faultevent.data import DataLoader
from faultevent.signal import Signal, SignalModel


class MLConfig(TypedDict):
    model_path: str


def model_filepath(dataset: data.DataName):
    model_path = pathlib.Path(load_config()["ml"]["model_path"])
    return model_path/f"{dataset}.pt"


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        depth = 5
        hidden_channels = 10
        kernel_size = 3
        stride = 2
        dilation = 1
        self.kernel_size = kernel_size
        self.stride = stride

        self.downsample = [torch.nn.Conv1d(1, hidden_channels, kernel_size,
                                      stride=stride, dilation=dilation)]
        self.downsample += [
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                            stride=stride, dilation=dilation)
            for _ in range(depth)
        ]
        self.upsample = [
            torch.nn.ConvTranspose1d(hidden_channels, hidden_channels,
                                     kernel_size, stride=stride,
                                     dilation=dilation)
            for _ in range(depth)
        ]
        self.upsample += [torch.nn.ConvTranspose1d(hidden_channels, 1,
                                                kernel_size, stride=stride,
                                                dilation=dilation)]
        self.activation = torch.nn.ELU()

        self.layers = torch.nn.ModuleList((*self.downsample, *self.upsample))

    def forward(self, x):
        # shape(x) = (N, 1, L)

        padlog = torch.zeros((len(self.downsample)), dtype=int)
        for i, layer in enumerate(self.downsample):
            lin = x.shape[-1]
            x = layer(x)
            lout = x.shape[-1]
            padlog[i] = lin//self.stride-lout
            x = self.activation(x)
        
        padlog = torch.flip(padlog, (0,))
        for i, layer in enumerate(self.upsample):
            x = layer(x)
            x = torch.nn.functional.pad(x, (0, padlog[i]))
            if i<len(self.upsample)-1:
                x = self.activation(x)

        return x
    
    @classmethod
    def load(cls, savepath: str):
        model = cls()
        state = torch.load(savepath)
        model.load_state_dict(state["model_state_dict"])
        return model


def augment_sequence(signal, snr: float):
    """We want to make the model agnostic to
        1. any specific fault frequency, and
        2. any specific signature waveform.
    Therefore, the signal should be augmented by white noise,
    occurring in pulses at random points in time.
    """
    pulse_length = 30

    tilde = np.zeros_like(signal)

    # snr = pow(signal) / pow(noise)
    # pow(signal) = snr * pow(noise)
    # std(signal) = sqrt[snr*pow(noise)]
    std_tilde = np.sqrt(snr*np.var(signal))

    idx = 0
    while idx <= len(signal):
        idx0 = idx + np.random.poisson(200)
        if idx0 > len(signal):
            break
        idx1 = min(idx0 + pulse_length, len(signal))
        pulse_length_actual = idx1 - idx0
        signature = np.random.randn(pulse_length_actual)*std_tilde
        tilde[idx0:idx1] = signature
        idx = idx1

    return signal + tilde


type PrepDataset = Iterator[np.NDArray[np.float32]]
type Dataset = np.NDArray[np.float32]


def prepare_dataset(dataloader: DataLoader, signal_ids: Sequence, siglen: int,
                    nval: int = 0, standardize = False) -> PrepDataset:
    """Prepare a dataset (in memory) from the dataloader using signals
    identified by the provided signal IDs. Returns an iterator of signals.
    """
    getsig = lambda idx: dataloader[idx].vib.y
    s = map(getsig, signal_ids)

    def split(s):
        remain = len(s)%siglen
        nsig = len(s)//siglen
        return np.split(s[:-remain], nsig)

    s = map(split, s)
    s = itertools.chain.from_iterable(s)
    s = np.array(list(s)) # to memory here

    if standardize:
        s -= np.mean(s)
        s /= np.std(s)

    if nval:
        idx_val = np.linspace(0, len(s)-1, nval, dtype=int)
        sval = s[idx_val]
        idx_train = np.setdiff1d(np.arange(len(s)), idx_val)
        strain = s[idx_train]

        return np.vstack(strain, dtype=np.float32), np.vstack(sval, dtype=np.float32)
    
    return np.vstack(s, dtype=np.float32)


def gen_batches(dataset: Dataset, batch_size: int, shuffle = True) -> Dataset:
    """Generate batches for one epoch of the dataset."""
    
    rng = np.random.default_rng()
    if shuffle:
        rng.shuffle(dataset, axis=0)

    yield from itertools.batched(dataset, batch_size)


def train(dataset_train: PrepDataset, 
          dataset_val: PrepDataset,
          batch_size: int, epochs: int, savepath: str,
          dataset_validate: PrepDataset = None,
          overwrite = False):
    
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epoch = 0
    loss_history = []
    loss_history_val = []
    if not overwrite:
        try:
            state = torch.load(savepath)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            epoch = state["epoch"]+1
            loss_history = state["loss_history"]
            loss_history_val = state["loss_history_val"]
        except Exception:
                print("could not load model state")

    loss_fn = torch.nn.MSELoss()

    range_epochs = range(epoch, epoch+epochs)

    for epoch in range_epochs:
        train_batches = gen_batches(dataset_train, batch_size, shuffle=True,)
        # epoch training pass
        model.train()
        for i, batch in enumerate(train_batches):
            
            #aug_batch = map(partial(augment_sequence, 10.0), batch)
            aug_batch = np.apply_along_axis(augment_sequence, -1, batch, snr=10.0)
            
            # (N, 1, L)
            desired_shape = (batch_size, 1, -1)
            batch = torch.from_numpy(np.reshape(batch, desired_shape))
            aug_batch = torch.from_numpy(np.reshape(aug_batch, desired_shape))

            optimizer.zero_grad()

            # model specifics
            pred_batch = model(aug_batch)

            loss = loss_fn(pred_batch, batch)
            loss.backward()

            optimizer.step()
            print(f"e{epoch},s{i}\t{loss.item()}")

        # epoch validation pass
        with torch.no_grad():
            test_batches = gen_batches(dataset_val, len(dataset_val), shuffle=True,)
            batch = next(test_batches)
            model.eval()
            aug_batch = np.apply_along_axis(augment_sequence, -1, batch, snr=10.0)
            desired_shape = (batch_size, 1, -1)
            batch = torch.from_numpy(np.reshape(batch, desired_shape))
            aug_batch = torch.from_numpy(np.reshape(aug_batch, desired_shape))
            pred_batch = model(aug_batch)
            loss_val = loss_fn(pred_batch, batch)
            loss_history_val.append(loss_val.item())

        
        loss_history.append(loss.item())
        # save model after every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "loss_history": loss_history,
            "loss_history_val": loss_history_val,
        }, savepath)
    

class MLSignalModel(SignalModel):

    def __init__(self, model: Model):
        self.model = model

    def process(self, signal: Signal) -> Signal:
        with torch.no_grad():
            bx = torch.tensor(signal.y, dtype=torch.float32).reshape((1, 1, -1))
            by = self.model(bx)
            y = torch.flatten(by).numpy()
            out = Signal(y, signal.x, uniform_samples=signal.uniform_samples)
            return out

    def residuals(self, signal: Signal) -> Signal:
        hest = self.process(signal)
        out = signal-hest
        return out


def plot_history(name: data.DataName):
    import matplotlib.pyplot as plt
    savepath = model_filepath(name)
    state = torch.load(savepath)
    epoch = state["epoch"]+1
    loss_history = state["loss_history"]
    loss_history_val = state["loss_history_val"]
    plt.figure()
    plt.plot(range(epoch), loss_history, label="train")
    plt.plot(range(epoch), loss_history_val, label="validate")
    plt.title(f"training history for\n\"{name}\" dataset")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    import argparse
    import glob

    cfg = load_config()

    # TODO: Remove `out` argument, add `overwrite` argument

    parser = argparse.ArgumentParser(
        prog="Fit signal model",
    )
    parser.add_argument("name", choices=list(data.DataName), help="name of the dataset")
    parser.add_argument("-x", "--overwrite", action="store_true",
                        help="overwrite existing model")
    parser.add_argument("-l", "--len", type=int, default=10000,
                        help="signal training length")
    parser.add_argument("-b", "--bsize", type=int, default=10,
                        help="batch size")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="number of epochs before termination")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dl = data.dataloader(args.name)
    data_path = data.data_path(args.name)
    signal_idx = []
    match type(dl):
        
        case data.UiADataLoader:
            exclude_idx = ["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
            signal_idx = filter(
                    lambda x: "1000rpm" in x
                    and not pathlib.Path(x) in map(pathlib.Path, exclude_idx),
                glob.iglob("y2016-m09-d20/*.h5", root_dir=data_path))
        
        case data.UNSWDataLoader:
            exclude_idx = ["Test 1/6Hz/vib_000002663_06.mat"]
            signal_idx = filter(
                    lambda x: not pathlib.Path(x) in map(pathlib.Path, exclude_idx),
                itertools.islice(
                    glob.iglob("Test 1/6Hz/*.mat", root_dir=data_path), 10))
        
        case data.CWRUDataLoader:
            exclude_idx = ["099"]
            signal_idx = ["097", "098", "100"] # "099" excluded
        case _:
            raise ValueError("dataloader not recognized")
    
    savepath = model_filepath(args.name)
    dataset = prepare_dataset(dataloader=dl, signal_ids=signal_idx,
                              siglen=args.len, nval=5, standardize=False)
    train(*dataset, args.bsize, args.epochs, savepath, overwrite=args.overwrite)
