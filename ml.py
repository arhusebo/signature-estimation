from typing import Sequence, TypedDict
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
        mean = torch.mean(x, dim=-1, keepdim=True)
        x -= mean
        std = torch.std(x, dim=-1, keepdim=True)
        x /= 3*std

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
        x *= 3*std
        x -= torch.mean(x, dim=-1, keepdim=True)
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


def batchify(x, siglen: int):
    # The number of batches is determined by the specified signal
    # length. The implications are that the batch size may vary 
    # depending on the length of the loaded signal.
    nbatches = len(x)//siglen
    # (N, L)
    xb = torch.tensor(list(itertools.batched(x[:nbatches*siglen], siglen)),
                        dtype=torch.float32)
    # (N, 1, L)
    xb = xb[:,torch.newaxis,:]

    return xb, nbatches


def train(dataloader: DataLoader, signal_idx: Sequence, epochs: int,
          siglen: int, savepath: str, overwrite = False):
    
    model = Model()
    
    epoch = 0
    if not overwrite:
        try:
            state = torch.load(savepath)
            model.load_state_dict(state["model_state_dict"])
            epoch = state["epoch"]+1
        except Exception:
                print("could not load model state")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    range_epochs = range(epoch, epoch+epochs)
    epoch_loss = []
    for epoch in range_epochs:
        # TODO: Split signals and batching
        for i, idx_ in enumerate(signal_idx):
            vib = dataloader[idx_].vib
            
            x = vib.y
            ax = augment_sequence(x, 10.0)

            optimizer.zero_grad()

            # model specifics
            xb, _ = batchify(x, siglen)
            axb, _ = batchify(ax, siglen)
            yb = model(axb)

            loss = loss_fn(yb, xb)
            loss.backward()

            optimizer.step()
            epoch_loss.append(loss)
            print(f"e{epoch},s{i}\t{loss.item()}")
        
        # save model after every epoch
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, savepath)
    

class MLSignalModel(SignalModel):

    def __init__(self, model: Model):
        self.model = model

    def process(self, signal: Signal) -> Signal:
        with torch.no_grad():
            bx, _ = batchify(signal.y, len(signal))
            by = self.model(bx)
            y = torch.flatten(by).numpy()
            out = Signal(y, signal.x, uniform_samples=signal.uniform_samples)
            return out

    def residuals(self, signal: Signal) -> Signal:
        hest = self.process(signal)
        out = signal-hest
        return out


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
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="number of epochs before termination")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    not_implemented = NotImplementedError("program not implemented for this dataset")

    dl = data.dataloader(args.name)
    data_path = data.data_path(args.name)
    signal_idx = []
    match type(dl):
        
        case data.UiADataLoader:
            exclude = ["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
            signal_idx = filter(
                    lambda x: "1000rpm" in x
                    and not pathlib.Path(x) in map(pathlib.Path, exclude),
                glob.iglob("y2016-m09-d20/*.h5", root_dir=data_path))
        
        case data.UNSWDataLoader:
            exclude = ["Test 1/6Hz/vib_000002663_06.mat"]
            signal_idx = filter(
                    lambda x: not pathlib.Path(x) in map(pathlib.Path, exclude),
                itertools.islice(
                    glob.iglob("Test 1/6Hz/*.mat", root_dir=data_path), 10))
        
        case data.CWRUDataLoader:
            signal_idx = ["097", "098", "100"] # "099" excluded
        case _:
            raise ValueError("dataloader not recognized")
    
    savepath = model_filepath(args.name)
    train(dl, list(signal_idx), args.epochs, args.len, savepath, overwrite=args.overwrite)