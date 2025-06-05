from typing import Sequence, Literal
import itertools
import numpy as np
import torch
import data

from faultevent.data import DataLoader

class RecurrentModel(torch.nn.Module):
    """This model is not fully developed and tested"""

    def __init__(self, depth = 5, hidden_size = 20):
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size

        self.rnn = torch.nn.RNN(1, self.hidden_size, self.depth)
        self.out = torch.nn.Linear(self.hidden_size, 1)
    
    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        x = self.out(x)
        return x


class ConvolutionalModel(torch.nn.Module):

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


def batchify(x, model_type: Literal["recu", "conv"], siglen: int):
    # The number of batches is determined by the specified signal
    # length. The implications are that the batch size may vary 
    # depending on the length of the loaded signal.
    nbatches = len(x)//siglen
    # (N, L)
    xb = torch.tensor(list(itertools.batched(x[:nbatches*siglen], siglen)),
                        dtype=torch.float32)
    match model_type:
        case "recu":
            # (L, N, 1)
            xb = xb.permute(-1, 0)[:,:,torch.newaxis]
        case "conv":
            # (N, 1, L)
            xb = xb[:,torch.newaxis,:]
        case _:
            raise ValueError("invalid model type")

    return xb, nbatches


def train(model: RecurrentModel | ConvolutionalModel, dataloader: DataLoader,
          signal_idx: Sequence, epochs: int, siglen: int, savepath: str):
    
    epoch = 0
    try:
        state = torch.load(savepath)
        model.load_state_dict(state["model_state_dict"])
        epoch = state["epoch"]
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
            match model:
                case t if isinstance(t, RecurrentModel):
                    xb, nbatches = batchify(x, "recu", siglen)
                    axb, _ = batchify(ax, "recu", siglen)
                    h0 = torch.randn(model.depth, nbatches, model.hidden_size,
                                    dtype=torch.float).to(device)
                    yb = model(axb, h0)
                case t if isinstance(t, ConvolutionalModel):
                    xb, nbatches = batchify(ax, "conv", siglen)
                    axb, _ = batchify(ax, "conv", siglen)
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
    


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        prog="Fit signal model",
    )
    parser.add_argument("name", help="name of the dataset")
    parser.add_argument("model", choices=["recu", "conv"], help="type of model")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="model output path")
    parser.add_argument("-l", "--len", type=int, default=10000,
                        help="signal training length")
    parser.add_argument("-e", "--epochs", type=int, default=20,
                        help="number of epochs before termination")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    not_implemented = NotImplementedError("program not implemented for this dataset")

    dl = data.dataloader(args.name)
    signal_idx = []
    match type(dl):
        case data.UiADataLoader:
            glb = str(dl.path/"y2016-m09-d20")+"/*.h5"
            signal_idx = [f for f in glob.glob(glb) if "1000rpm" in f]
        case data.UNSWDataLoader:
            glb = str(dl.path/"Test 1/6Hz")+"/*.mat"
            signal_idx = glob.glob(glb)[:10]
        case data.CWRUDataLoader:
            signal_idx = ["097", "098", "099", "100"]
            # raise not_implemented
        case _:
            raise ValueError("dataloader not recognized")

    match args.model:
        case "recu":
            model = RecurrentModel()
        case "conv":
            model = ConvolutionalModel()
        case _:
            raise ValueError("invalid model type")
    
    savepath = args.out
    if savepath is None:
        savepath = f"./{args.name}_{args.model}.pt"

    train(model, dl, signal_idx, args.epochs, args.len, savepath)