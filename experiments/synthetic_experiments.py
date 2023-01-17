import numpy as np
import scipy.linalg

from faultevent.signal import Signal
import routines

import gsim
from gsim.gfigure import GFigure


def generate_resid(signat, snr, siglen, fs, fss, ordf, seed=0):
    rng = np.random.default_rng(seed)
    df = (fs/fss)/ordf
    sigloc = np.arange(0, siglen, df, dtype=float)
    sigloc += rng.standard_normal(len(sigloc))*50
    signal = np.sum([signat(np.arange(siglen)-n0) for n0 in sigloc], axis=0)
    sigpow = np.var(signal)
    noisepow = sigpow/snr
    noise = rng.standard_normal(siglen) * np.sqrt(noisepow)
    resid = signal + noise
    out = Signal.from_uniform_samples(resid, 1/(fs/fss))
    return out

def estimate_signat(resid: Signal, indices, siglen, shift):
    n_samples = 0
    running_sum = np.zeros((siglen,))
    for idx in indices:
        if idx+shift >= 0 and idx+shift+siglen < len(resid):
            n_samples += 1
            running_sum += resid.y[idx+shift:idx+shift+siglen]
    sig = running_sum/n_samples
    return sig

def _rmse_shift(sigest, sigtruefunc, shiftmax=100):
    siglen = len(sigest)
    sigestpad = np.pad(sigest, shiftmax)
    sigestpad /= np.linalg.norm(sigestpad)
    sigtrue = sigtruefunc(np.arange(siglen))
    sigtrue /= np.linalg.norm(sigtrue)
    rmse = np.linalg.norm([sigestpad[i:i+siglen] - sigtrue for i in range(2*shiftmax)], axis=-1)
    n = np.arange(2*shiftmax)-shiftmax
    return rmse, n

def estimate_rmse(sigest, sigtruefunc, shiftmax=100):
    rmse, n = _rmse_shift(sigest, sigtruefunc, shiftmax)
    idxmin = np.argmin(rmse)
    return rmse[idxmin], n[idxmin]


def general_snr_experiment(sigfunc):

    # synthethic data generator parameters
    seed = 0
    siglen = 100000 # length of synthetic signal
    fs = 51200 # virtual sample frequency
    fss = 1000/60 # shaft frequency
    ordf = 5.0 # fault order
    avg_event_period = (fs/fss)/ordf # in samples

    snr_to_eval = np.logspace(-2, 0, 10)

    sigestlen = 200
    sigestshift = -20

    rmse = np.zeros((3, len(snr_to_eval)), dtype=float)

    for i, snr in enumerate(snr_to_eval):
        # generate residuals at given snr
        resid = generate_resid(sigfunc, snr, siglen, fs, fss, ordf, seed=seed)

        # estimate signature using IRFS
        sigest_irfs = routines.irfs(resid, 4.5, 5.5,
                                    sigsize=sigestlen,
                                    sigshift=sigestshift,)
        
        # estimate signature using MED and peak detection
        medout = routines.medfilt(resid, 100)
        medenv = abs(scipy.signal.hilbert(medout.y))
        medpeaks, _ = scipy.signal.find_peaks(medenv, distance=avg_event_period/2)
        sigest_med = estimate_signat(resid, medpeaks, sigestlen, sigestshift)
        
        # estimate signature using SK and peak detection
        skout = routines.skfilt(resid, 100)
        skenv = abs(skout.y)
        skpeaks, _ = scipy.signal.find_peaks(skenv, distance=avg_event_period/2)
        sigest_sk = estimate_signat(resid, skpeaks, sigestlen, sigestshift)

        rmse[0, i], _ = estimate_rmse(sigest_irfs, sigfunc)
        rmse[1, i], _ = estimate_rmse(sigest_med, sigfunc)
        rmse[2, i], _ = estimate_rmse(sigest_sk, sigfunc)

    legend = ["IRFS", "MED", "SK"]

    G = GFigure(xaxis=snr_to_eval,
                yaxis=rmse,
                legend=legend,
                xlabel="SNR (dB)",
                ylabel="RMSE")
    return G

class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
        """Experiment for testing RMSE estimation"""

        siglen = 100
        shift = 50
        sigtrue = lambda n: (n<10)*(n>=0)*1.0
        sigest = np.zeros((siglen,), dtype=float)
        sigest[shift:shift+10] = 1.0
        sigest += np.random.randn(siglen)*.01
        shiftmax = 100
        rmse, n = _rmse_shift(sigest, sigtrue, shiftmax)
        
        G = GFigure(xaxis=n, yaxis=rmse)
        return G

    def experiment_1002(l_args):
        """Benchmark methods under varying SNR for signature type A"""

        sigfunc = lambda n: (n>=0)*np.sinc(n/8+1)
        G = general_snr_experiment(sigfunc)
        return G

    def experiment_1003(l_args):
        """Benchmark methods under varying SNR for signature type B"""

        sigfunc = lambda n: (n>=0)*np.sinc(n/2+1)
        G = general_snr_experiment(sigfunc)
        return G
    
    def experiment_1004(l_args):
        """Benchmark methods under varying SNR for signature type C"""

        sigfunc = lambda n: (n>=0)*np.exp(-abs(n)/10)
        G = general_snr_experiment(sigfunc)
        return G
    
    def experiment_1005(l_args):
        """Benchmark methods under varying SNR for signature type D"""

        sigfunc = lambda n: (n<10)*(n>=0)*1.0
        G = general_snr_experiment(sigfunc)
        return G

    def experiment_1006(l_args):
        print("Combining previous experiments")

        l_G2 = ExperimentSet.load_GFigures(1002)[0]
        l_G3 = ExperimentSet.load_GFigures(1003)[0]
        l_G4 = ExperimentSet.load_GFigures(1004)[0]
        l_G5 = ExperimentSet.load_GFigures(1005)[0]
        
        G = GFigure(figsize=(5.5, 10.0))
        G.l_subplots = l_G2.l_subplots +\
                       l_G3.l_subplots +\
                       l_G4.l_subplots +\
                       l_G5.l_subplots
        return G