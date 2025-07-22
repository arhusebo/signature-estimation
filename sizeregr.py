import pathlib
import numpy as np
from scipy.signal import iirfilter, lfilter
import torch

from config import load_config

INPUT_LENGTH = 100 # signature length for model input

def model_filepath():
    model_path = pathlib.Path(load_config()["ml"]["model_path"])
    return model_path/"sizeregr.pt"


class ModelDNN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        width = 100
        depth = 10

        self.input = torch.nn.Linear(400,width)
        self.hidden = [torch.nn.Linear(width, width)]*depth
        self.output = torch.nn.Linear(width,1)
        self.activation = torch.nn.ReLU()
        
        self.layers = torch.nn.ModuleList((self.input, *self.hidden, self.output))
    
    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.output(x)

    @classmethod
    def load(cls, savepath: str):
        model = cls()
        state = torch.load(savepath)
        model.load_state_dict(state["model_state_dict"])
        return model


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        channels = 5
        conv_depth = 3
        kernel = 3
        stride = 1

        depth = 3
        width = 50

        self.input = torch.nn.Conv1d(1, channels, kernel, stride)
        self.bnorm_in = torch.nn.BatchNorm1d(channels)
        self.conv_hidden = [torch.nn.LazyConv1d(channels, kernel, stride)]*(conv_depth-1)
        self.conv_out = torch.nn.LazyConv1d(1, kernel, stride)
        self.hidden = [torch.nn.LazyLinear(width)]+[torch.nn.Linear(width, width)]*depth
        self.output = torch.nn.LazyLinear(1)
        self.act_conv = torch.nn.ReLU()
        self.act_linr = torch.nn.Tanh()
        self.pool = torch.nn.MaxPool1d(2)
        self.bnorm_hidden = [torch.nn.BatchNorm1d(channels)]*len(self.conv_hidden)
        
        self.layers = torch.nn.ModuleList((self.input, self.bnorm_in,
                                           *self.conv_hidden,
                                           *self.bnorm_hidden,
                                           *self.hidden, self.output))
    
    def forward(self, x):
        x = self.input(x)
        x = self.bnorm_in(x)
        x = self.act_conv(x)
        for conv, bnorm in zip(self.conv_hidden, self.bnorm_hidden):
            x = conv(x)
            x = bnorm(x)
            x = self.act_conv(x)
            x = self.pool(x)
        x = self.act_conv(self.conv_out(x))
        x = x.view(x.size(0), -1)
        for layer in self.hidden:
            x = layer(x)
            x = self.act_linr(x)
        return self.output(x)
    
    @classmethod
    def load(cls, savepath: str):
        model = cls()
        state = torch.load(savepath)
        model.load_state_dict(state["model_state_dict"])
        return model


def signature_input(fsize: int, shift: int, n: int):
    return -0.1*(shift<n<shift+fsize) + 1.0*(n==shift+fsize)


vec_signature_input = np.vectorize(signature_input)


def filter_input(x):
    b, a = iirfilter(4, (0.4, 0.6),)
    return lfilter(b, a, x)


def signature_factory(fsize: int, shift: int, length: int):
    n = np.arange(length)
    x = vec_signature_input(fsize, shift, n)
    y = filter_input(x)
    return y.astype(np.float32)


def samples2fun(y):
    return lambda n: y[n] if len(y)>n>=0 else 0.0


def create_batch(size: int, noise_std=0.0):
    n = torch.arange(0, INPUT_LENGTH)
    labels = np.zeros(size, dtype=np.float32)
    batch = np.zeros((size, len(n)), dtype=np.float32)
    for i in range(size):
        fsize = np.random.randint(5, 30)
        shift = np.random.randint(10, 25)
        siginp = vec_signature_input(fsize, shift, n)
        batch[i] = filter_input(siginp)+np.random.randn(len(siginp))*noise_std
        labels[i] = fsize
    
    return batch[:,np.newaxis,:], labels


def pred(model, fault_signature):
    s = fault_signature[torch.newaxis,torch.newaxis,:INPUT_LENGTH]
    return model(s)

def pred_np(model, fault_signature):
    return pred(model, torch.from_numpy(fault_signature)).detach().numpy().item()

def pred_err_np(fsize, model, fault_signature):
    return abs(pred(model, torch.from_numpy(fault_signature))
               .detach().numpy().item()-fsize)

def mae_loss(x, y):
    return torch.median(torch.abs(x-y)**2)

def train(iterations: int, savepath: str, save_every: int = None, overwrite = False):
    
    model = Model()
    
    iteration = 0
    if not overwrite:
        try:
            state = torch.load(savepath)
            model.load_state_dict(state["model_state_dict"])
            iteration = state["iter"]+1
        except Exception:
                print("could not load model state")

    # loss_fn = torch.nn.MSELoss()
    loss_fn = mae_loss
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    range_iter = range(iteration, iteration+iterations)
    iter_loss = []
    for i, iter in enumerate(range_iter):
    # TODO: Split signals and batching
        npx, npy = create_batch(100, 0.01)
        x = torch.tensor(npx)
        y = torch.tensor(npy)

        optimizer.zero_grad()
        ye = model(x)

        loss = loss_fn(ye[:,0], y)
        loss.backward()

        optimizer.step()
        iter_loss.append(loss)
        print(f"i{iter},s{i}\t{loss.item()}")
    
        # save model after every epoch
        savedict = {
                "iter": iter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
        if not save_every is None and i%save_every==0:
            torch.save(savedict, savepath)
        
    if save_every is None:
        torch.save(savedict, savepath)


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(
        prog="Fit signal model",
    )
    parser.add_argument("-x", "--overwrite", action="store_true",
                        help="overwrite existing model")
    parser.add_argument("-i", "--iter", type=int, default=20,
                        help="number of iterations before termination")
    parser.add_argument("-c", "--checkpoint", type=int, default=None,
                        help="every n-th iteration to save the model")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(args.iter, model_filepath(), save_every=args.checkpoint,
          overwrite=args.overwrite)