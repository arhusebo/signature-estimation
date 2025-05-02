import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from faultevent.signal import Signal, ARModel
from data.uia import UiADataLoader
from data import uia_path

t_end = 0.02
fs = 51200
df = 0.005

t = np.arange(0, t_end, 1/fs)

dl = UiADataLoader(uia_path)
mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
model = ARModel.from_signal(mh.vib[:10000], 117) # AR model

siginp = np.zeros((5,))
siginp[:4] = -5.0
siginp[4] = 10.0

sig_ = np.zeros_like(t)
sig_[:len(siginp)] = siginp
# sig_[10] = 1.0
sig_ = Signal.from_uniform_samples(sig_, 1/fs)
sig = model.process(sig_)

tf = np.arange(0, t_end, df)

sigtilde_ = np.zeros_like(t)
for t_ in tf:
    idx = int(t_*fs)
    sigtilde_[idx:idx+len(siginp)] = siginp
sigtilde_ = Signal.from_uniform_samples(sigtilde_, 1/fs)
sigtilde = model.process(sigtilde_)


h0_ = Signal.from_uniform_samples(np.random.randn(len(t)), 1/fs)
h0 = model.process(h0_)



matplotlib.rcParams.update({"font.size": 6})


fig, ax = plt.subplots(4, 1, sharex=False, figsize=(3.5, 1.5))
ax[0].plot(sig.x, sig.y, c="k", lw=0.5)
ax[0].set_ylabel(r"$s(t)$")

ax[1].plot(sigtilde.x, sigtilde.y, c="k", lw=0.5)
ax[1].set_ylabel(r"$\tilde{s}(t)$")

ax[2].plot(h0.x, h0.y, c="k", lw=0.5)
ax[2].set_ylabel(r"$\epsilon(t)$")

ax[3].plot(h0.x, h0.y+sigtilde.y, c="k", lw=0.5)
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