"""ml2.py -- a second denoiser that directly estimates the *non-healthy*
(fault signature train) component of a vibration signal.

Unlike `ml.py` (trained self-supervised to reconstruct the healthy signal,
exposing the fault only through the residual), this model is trained fully
*supervised* on synthetic signals from the same signal model as
`experiments.synth.ex_signature_recovery`: the input is `healthy +
signature-train` and the target is the signature train itself.

The healthy components used for training are held out from the recording used
in `ex_signature_recovery`, so the test healthy signal is never seen during
training. A fresh signal is drawn for every example, so there is no need for a
validation split. SNR is drawn log-uniformly over `SNR_DB_RANGE`.

Train:  python -m ml2 unsw -s 4000        # -> models/unsw_ml2.pt
"""

import argparse
import glob
import itertools
import pathlib

import numpy as np
import torch
import torch.nn as nn

import data
import data.synth as synth
from faultevent.signal import Signal, SignalModel
from config import load_config


# --- signal-model parameters (mirror experiments.synth.make_signal_config) ---
FS = 51200                 # sample frequency
FSHAFT = 1000 / 60         # shaft frequency
ORDF = 5.0                 # fault order
FAULT_STD = 0.01           # event shaft-position jitter
SIG_F, SIG_TAU, SIG_FS = 6.5e3, 0.001, 25.e3
SIG_LEN = 800
FSIZE_INTERVAL = (10, 40)
SNR_DB_RANGE = (-30.0, 0.0)


def model_filepath(dataname: data.DataName) -> pathlib.Path:
    return pathlib.Path(load_config()["ml"]["model_path"]) / f"{dataname}_ml2.pt"


# --- architecture ------------------------------------------------------------
class Denoiser(nn.Module):
    """Fully-convolutional, same-length dilated residual CNN mapping a noisy
    1-channel signal to an estimate of its fault (non-healthy) component.

    The dilation schedule gives a receptive field of ~1000 samples (a couple
    of fault periods). Because the target is the *sparse* fault component,
    there is no identity-map shortcut for the network to fall into (unlike a
    healthy-signal reconstructor), so it is forced to learn to extract the
    fault. Symmetric padding keeps the output aligned with the input."""

    def __init__(self, channels=32, kernel=9, dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = nn.Conv1d(1, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, dilation=d,
                      padding=d * (kernel - 1) // 2)
            for d in dilations)
        self.out = nn.Conv1d(channels, 1, 1)
        self.act = nn.ELU()

    def forward(self, x):
        h = self.act(self.inp(x))
        for block in self.blocks:
            h = h + self.act(block(h))   # residual refinement in feature space
        return self.out(h)


# --- synthetic training data (in-memory; mirrors generate_vibration) ---------
def _fault_signature(rng):
    fsize = rng.integers(*FSIZE_INTERVAL)
    return synth.signt_res(SIG_F, SIG_TAU, fsize, np.arange(SIG_LEN), fs=SIG_FS)


def synth_example(rng, noise_pool, length):
    """Build one (input, target) pair from the same signal model used by
    `generate_vibration`, with the healthy floor drawn from `noise_pool` and
    the fault SNR drawn log-uniformly over `SNR_DB_RANGE`. `target` is the
    non-healthy (signature-train) component."""
    noise_full = noise_pool[rng.integers(len(noise_pool))]
    idx0 = rng.integers(len(noise_full) - length)
    noise = noise_full[idx0:idx0 + length]
    pow_noise = np.var(noise)

    signature = _fault_signature(rng)
    snr = 10.0 ** (rng.uniform(*SNR_DB_RANGE) / 10.0)

    eosp_end = length / FS * FSHAFT
    eosp = np.arange(0, eosp_end, 1 / ORDF)
    eosp = eosp + rng.standard_normal(len(eosp)) * FAULT_STD
    component = synth.signature_train(eosp, signature, length, fs=FS, fshaft=FSHAFT)
    component = np.sqrt(pow_noise * snr) * component / np.std(signature)

    return noise + component, component


def synth_batch(rng, noise_pool, length, batch_size):
    xs = np.empty((batch_size, length), dtype=np.float32)
    ts = np.empty((batch_size, length), dtype=np.float32)
    for i in range(batch_size):
        xs[i], ts[i] = synth_example(rng, noise_pool, length)
    return xs, ts


# --- loss: scale-invariant NMSE (= 1 - rho^2) + a mild scale (MSE) term ------
def si_nmse(out, target, eps=1e-8):
    """Per-example scale-invariant NMSE, 1 - rho^2 (matches recovery_error)."""
    o = out.flatten(1)
    t = target.flatten(1)
    dot = (o * t).sum(1)
    rho2 = dot * dot / ((o * o).sum(1) * (t * t).sum(1) + eps)
    return (1.0 - rho2).mean()


def loss_fn(out, target, scale_weight=0.5):
    # SI term drives waveform shape (equal weight across SNR); the MSE term
    # calibrates amplitude (matters mostly at higher SNR, where the target is
    # not tiny) so the estimate is also reasonable on a unit-gain metric.
    return si_nmse(out, target) + scale_weight * torch.mean((out - target) ** 2)


# --- training ----------------------------------------------------------------
def train(noise_pool, savepath, steps=4000, batch_size=16, length=8192,
          lr=1e-3, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    model = Denoiser().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    hist = []
    for step in range(steps):
        xs, ts = synth_batch(rng, noise_pool, length, batch_size)
        x = torch.from_numpy(xs).unsqueeze(1).to(device)      # (B, 1, L)
        t = torch.from_numpy(ts).unsqueeze(1).to(device)
        sd = x.std(dim=-1, keepdim=True) + 1e-8               # per-example scale
        opt.zero_grad()
        out = model(x / sd)
        loss = loss_fn(out, t / sd)
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"step {step + 1}/{steps}  loss {np.mean(hist[-100:]):.4f}",
                  flush=True)
    torch.save({"model_state_dict": model.state_dict()}, savepath)
    print(f"saved {savepath}", flush=True)
    return model


# --- inference wrapper -------------------------------------------------------
class ML2SignalModel(SignalModel):
    """Wraps a trained `Denoiser`. `residuals` returns the estimated
    non-healthy (fault) component; `process` returns the implied healthy part
    (`signal - residuals`), so the class satisfies the `SignalModel` contract
    and plugs into `ex_signature_recovery` like the AR/ML models."""

    def __init__(self, model: Denoiser):
        self.model = model.eval()

    def _estimate(self, signal: Signal) -> np.ndarray:
        with torch.no_grad():
            device = next(self.model.parameters()).device
            y = np.asarray(signal.y, dtype=np.float32)
            sd = float(np.std(y)) or 1.0
            x = torch.from_numpy(y / sd).reshape(1, 1, -1).to(device)
            return self.model(x).flatten().cpu().numpy() * sd

    def residuals(self, signal: Signal) -> Signal:
        return Signal(self._estimate(signal), signal.x,
                      uniform_samples=signal.uniform_samples)

    def process(self, signal: Signal) -> Signal:
        return Signal(signal.y - self._estimate(signal), signal.x,
                      uniform_samples=signal.uniform_samples)


def load_model(dataname: data.DataName = data.DataName.UNSW) -> ML2SignalModel:
    model = Denoiser()
    state = torch.load(model_filepath(dataname), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return ML2SignalModel(model)


# --- held-out healthy pool ---------------------------------------------------
def load_noise_pool(dataname: data.DataName, n_files: int = 10):
    """Load healthy signals to draw the noise floor from during training,
    excluding the recording used as the test healthy component in
    `ex_signature_recovery`."""
    from experiments.synth import SIGNAL_ID_MAP
    test_id = SIGNAL_ID_MAP[dataname]
    dl = data.dataloader(dataname)
    dp = data.data_path(dataname)
    match dataname:
        case data.DataName.UNSW:
            candidates = itertools.islice(
                glob.iglob("Test 1/6Hz/*.mat", root_dir=dp), n_files + 1)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ml2",
                                description="train the direct fault-component denoiser")
    p.add_argument("name", nargs="?", default="unsw", choices=list(data.DataName))
    p.add_argument("-s", "--steps", type=int, default=4000)
    p.add_argument("-b", "--batch", type=int, default=16)
    p.add_argument("-l", "--length", type=int, default=8192)
    args = p.parse_args()

    dataname = data.DataName(args.name)
    device = pick_device()
    print(f"device: {device}")
    pool = load_noise_pool(dataname)
    train(pool, model_filepath(dataname), steps=args.steps,
          batch_size=args.batch, length=args.length, device=device)
