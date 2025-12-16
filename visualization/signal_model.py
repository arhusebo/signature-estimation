import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from faultevent.signal import Signal, ARModel
import data
import data.synth
from config import load_config

t_end = 0.02
fs = 51200
df = 0.005

t = np.arange(0, t_end, 1/fs)

cfg = load_config()

dl = data.dataloader(data.DataName.UNSW)
#mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
mh = dl["Test 1/6Hz/vib_000002663_06.mat"]

sig_f = 6.5e3
sig_tau = 0.001
sig_fs = 25.e3
fsize = 20
sig_t = np.arange(800)
signature = data.synth.signt_res(sig_f, sig_tau, fsize, sig_t, fs=sig_fs)

tf = np.arange(0, t_end, df)
idx_signature = np.array(tf*fs, dtype=int)

sigtilde = np.zeros_like(t)
for idx in idx_signature:
    idxmax = min(len(sigtilde), idx+len(signature))
    print(idx, idxmax)
    sigtilde[idx:idxmax] = signature[:idxmax-idx]
sigtilde = Signal.from_uniform_samples(sigtilde, 1/fs)
#sigtilde = Signal.from_uniform_samples(impulse_train, 1/fs)

sig = np.zeros_like(t)
sig[:len(signature)] = signature
sig = Signal.from_uniform_samples(sig, 1/fs)

h0_ = Signal.from_uniform_samples(np.random.randn(len(t)), 1/fs)

h0 = mh.vib[:len(t)]

matplotlib.rcParams.update({"font.size": 6})


fig, ax = plt.subplots(4, 1, sharex=False, figsize=(3.5, 1.5))
ax[0].plot(sig.x, sig.y, c="k", lw=0.5)
ax[0].set_ylabel(r"$s(t)$")

ax[1].plot(sigtilde.x, sigtilde.y, c="k", lw=0.5)
ax[1].set_ylabel(r"$x(t)$")

ax[2].plot(h0.x, h0.y, c="k", lw=0.5)
ax[2].set_ylabel(r"$\epsilon(t)$")

ax[3].plot(h0.x, h0.y+0.05*sigtilde.y, c="k", lw=0.5)
ax[3].set_ylabel(r"$y(t)$")


for i in range(4):
    ax[i].set_yticks([])
    ax[i].set_xticks([])
    ax[i].set_xticklabels([])
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['bottom'].set_visible(False)
    ax[i].spines['left'].set_visible(False)

ax[-1].set_xticks(tf)
ax[-1].set_xticklabels([f"$t_{n+1}$" for n, _ in enumerate(tf)])
# ax[-1].set_xlabel("Time")

plt.tight_layout()

plt.show()
