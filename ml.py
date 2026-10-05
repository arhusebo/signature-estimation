"""ml.py -- a nonlinear, shape-free predictor of the *healthy* component of a
vibration signal.

The model is trained on healthy recordings only: no synthetic faults are
injected, so it learns nothing about what a fault looks like. Like the AR
model, it predicts every sample from the samples before it (a causal
one-step predictor), and the fault is exposed through the prediction
residual: whatever the healthy dynamics cannot predict is left in the
residual with its real shape. Unlike the AR model, the predictor is
nonlinear and looks ~1000 samples back.

The healthy recordings used for training exclude the recording used in
`ex_signature_recovery`, and the last recording of the pool is held out for
validation; the learning-rate schedule follows the validation loss.

Train:  python -m ml unsw -s 4000         # -> models/unsw_ml_healthy.pt
"""

import argparse
import glob
import itertools
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from faultevent.signal import Signal, SignalModel
from config import load_config


def model_filepath(dataname: data.DataName) -> pathlib.Path:
    return pathlib.Path(load_config()["ml"]["model_path"]) / f"{dataname}_ml_healthy.pt"


# --- architecture ------------------------------------------------------------
class Predictor(nn.Module):
    """Causal dilated residual CNN mapping a 1-channel signal to a one-step
    prediction of itself: output sample n depends only on input samples
    before n. Same trunk as `ml2.Denoiser`, but with left-only padding and the
    input delayed by one sample, so the network cannot copy the sample it
    predicts (the identity map that makes a plain healthy-signal
    reconstructor useless). The receptive field is `receptive_field` past
    samples."""

    def __init__(self, channels=32, kernel=9, dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=d)
            for d in dilations)
        self.pads = [d * (kernel - 1) for d in dilations]
        self.out = nn.Conv1d(channels, 1, 1)
        self.act = nn.ELU()

    @property
    def receptive_field(self) -> int:
        return 1 + sum(self.pads)

    def forward(self, x):
        x = F.pad(x, (1, 0))[..., :-1]         # delay: sample n sees x[:n]
        h = self.act(self.inp(x))
        for pad, block in zip(self.pads, self.blocks):
            h = h + self.act(block(F.pad(h, (pad, 0))))   # causal residual block
        return self.out(h)


# --- training data: random windows of healthy recordings ---------------------
def healthy_batch(rng, pool, length, batch_size):
    xs = np.empty((batch_size, length), dtype=np.float32)
    for i in range(batch_size):
        x = pool[rng.integers(len(pool))]
        idx0 = rng.integers(len(x) - length)
        xs[i] = x[idx0:idx0 + length]
    return xs


# --- loss: one-step prediction error ------------------------------------------
def loss_fn(out, x, warmup):
    """Mean squared one-step prediction error, skipping the first `warmup`
    samples whose prediction relies on zero padding. On unit-variance input
    this is the normalised prediction-error variance (1 = no better than
    predicting zero)."""
    return torch.mean((out - x)[..., warmup:] ** 2)


def _batch_loss(model, xs, device):
    x = torch.from_numpy(xs).unsqueeze(1).to(device)      # (B, 1, L)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-8)          # per-example scale
    return loss_fn(model(x), x, model.receptive_field)


# --- training ----------------------------------------------------------------
def train(pool, savepath, steps=4000, batch_size=16, length=8192,
          lr=1e-3, seed=0, device="cpu",
          overwrite=False, sched_every=100, ckpt_every=500, min_lr=0.0,
          n_val=64):
    """Trains on all recordings in `pool` but the last, which is held out for
    validation (with a single recording, windows of it are used for both)."""
    model = Predictor().to(device)
    if length <= 2 * model.receptive_field:
        raise ValueError(f"length must exceed twice the receptive field "
                         f"({model.receptive_field})")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # stepped once per `sched_every` steps on the validation loss
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    pool_train = pool[:-1] if len(pool) > 1 else pool
    pool_val = pool[-1:]
    # fixed validation windows, from a stream separate from the training one
    xs_val = healthy_batch(np.random.default_rng([seed, 2**32 - 1]), pool_val,
                           length, n_val)

    def validate():
        model.eval()
        with torch.no_grad():
            loss = np.mean([_batch_loss(model, xs_val[i:i + batch_size], device).item()
                            for i in range(0, n_val, batch_size)])
        model.train()
        return float(loss)

    def save():
        # write to a temporary file first so an interrupted save cannot
        # corrupt the existing checkpoint
        tmppath = pathlib.Path(f"{savepath}.tmp")
        torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "lrs_state_dict": lrs.state_dict(),
                "loss_history": hist_loss,
                "val_history": hist_val,
                "lr_history": hist_lr,
            }, tmppath)
        tmppath.replace(savepath)

    step = 0
    hist_loss = []
    hist_val = []       # (step, validation loss)
    hist_lr = []
    if not overwrite and pathlib.Path(savepath).exists():
        state = torch.load(savepath, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        opt.load_state_dict(state["optimizer_state_dict"])
        lrs.load_state_dict(state["lrs_state_dict"])
        step = state["step"]
        hist_loss = state["loss_history"]
        hist_val = state["val_history"]
        hist_lr = state["lr_history"]
        print(f"resuming from {savepath} at step {step}", flush=True)
    else:
        print("training from scratch", flush=True)
    # seeded by the global step so a resumed run draws new windows instead of
    # replaying the ones already trained on
    rng = np.random.default_rng([seed, step])
    model.train()

    try:
        for step_ in range(steps):
            xs = healthy_batch(rng, pool_train, length, batch_size)
            opt.zero_grad()
            loss = _batch_loss(model, xs, device)
            loss.backward()
            opt.step()
            step += 1
            hist_loss.append(loss.item())
            if (step_ + 1) % sched_every == 0:
                loss_val = validate()
                hist_val.append((step, loss_val))
                lrs.step(loss_val)
                print(f"step {step_ + 1}/{steps}  loss {np.mean(hist_loss[-sched_every:]):.4f}"
                      f"  val {loss_val:.4f}  lr {lrs.get_last_lr()[0]:.2e}", flush=True)
            hist_lr.append(lrs.get_last_lr())

            if (step_ + 1) % ckpt_every == 0:
                save()
            if opt.param_groups[0]["lr"] < min_lr:
                print(f"stopping early at step {step_ + 1}: lr below {min_lr:g}",
                      flush=True)
                break
    except KeyboardInterrupt:
        print("Training was interrupted early...")

    save()
    print(f"saved {savepath}", flush=True)
    return model


# --- inference wrapper -------------------------------------------------------
class MLSignalModel(SignalModel):
    """Wraps a trained `Predictor`. `process` returns the one-step prediction
    of the healthy component; `residuals` returns the prediction error
    (`signal - process`), which, like the AR residual, holds whatever the
    healthy dynamics cannot predict. The first `receptive_field` samples are
    dropped from the residual, as the AR model drops its first `p`."""

    def __init__(self, model: Predictor):
        self.model = model.eval()

    def process(self, signal: Signal) -> Signal:
        with torch.no_grad():
            device = next(self.model.parameters()).device
            y = np.asarray(signal.y, dtype=np.float32)
            sd = float(np.std(y)) or 1.0
            x = torch.from_numpy(y / sd).reshape(1, 1, -1).to(device)
            pred = self.model(x).flatten().cpu().numpy() * sd
        return Signal(pred, signal.x, uniform_samples=signal.uniform_samples)

    def residuals(self, signal: Signal) -> Signal:
        n = self.model.receptive_field
        e = signal.y[n:] - self.process(signal).y[n:]
        return Signal(e, signal.x[n:], uniform_samples=signal.uniform_samples)


def load_model(dataname: data.DataName = data.DataName.UNSW) -> MLSignalModel:
    model = Predictor()
    state = torch.load(model_filepath(dataname), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return MLSignalModel(model)


# --- held-out healthy pool ---------------------------------------------------
def load_noise_pool(dataname: data.DataName, n_files: int = 10):
    """Load healthy signals to train on, excluding the recording used as the
    test healthy component in `ex_signature_recovery`."""
    from experiments.synth import SIGNAL_ID_MAP
    test_id = SIGNAL_ID_MAP[dataname]
    dl = data.dataloader(dataname)
    dp = data.data_path(dataname)
    match dataname:
        case data.DataName.UNSW:
            candidates = itertools.islice(
                glob.iglob("Test 1/6Hz/*.mat", root_dir=dp), n_files + 1)
        case data.DataName.UIA:
            exclude_idx = ["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
            candidates = filter(
                    lambda x: "1000rpm" in x
                    and not pathlib.Path(x) in map(pathlib.Path, exclude_idx),
                glob.iglob("y2016-m09-d20/*.h5", root_dir=dp))
        case data.DataName.CWRU:
            exclude_idx = ["099"]
            candidates = ["097", "098", "100"] # "099" excluded
        case _:
            raise NotImplementedError(f"noise pool not defined for {dataname}")
    ids = [c for c in candidates
           if pathlib.Path(c) != pathlib.Path(test_id)][:n_files]
    pool = [np.asarray(dl[i].vib.y, dtype=np.float64) for i in ids]
    print(f"loaded {len(pool)} held-out healthy signals:")
    for i in ids:
        print("   ", i)
    return pool


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def plot_history(name: data.DataName):
    """Plot the model training history for a dataset"""
    import matplotlib.pyplot as plt
    savepath = model_filepath(name)
    state = torch.load(savepath, map_location="cpu")
    step = state["step"]
    loss_history = state["loss_history"]
    val_steps, val_loss = zip(*state["val_history"]) if state["val_history"] else ((), ())
    lr_history = np.ravel(state["lr_history"])

    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()
    ax.plot(range(step), loss_history, label="loss")
    ax.plot(val_steps, val_loss, label="validation loss")
    ax2.plot(range(step), lr_history, label="lr", color="k", ls="--")
    ax2.set_yscale("log")
    plt.title(f"training history for\n\"{name}\" dataset")
    fig.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax2.set_ylabel("lr")
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ml",
                                description="train the healthy-signal predictor")
    p.add_argument("name", nargs="?", default="unsw", choices=list(data.DataName))
    p.add_argument("-s", "--steps", type=int, default=4000)
    p.add_argument("-b", "--batch", type=int, default=16)
    p.add_argument("-l", "--length", type=int, default=8192)
    p.add_argument("-x", "--overwrite", action="store_true",)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=0.0,
                   help="stop early once the learning rate drops below this")
    args = p.parse_args()

    dataname = data.DataName(args.name)
    device = pick_device()
    print(f"device: {device}")
    pool = load_noise_pool(dataname)
    train(pool, model_filepath(dataname), steps=args.steps,
          batch_size=args.batch, length=args.length, device=device,
          overwrite=args.overwrite, lr=args.lr, min_lr=args.min_lr)
