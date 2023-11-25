import numpy as np
import scipy.linalg

from faultevent.signal import Signal
import algorithms

import gsim
from gsim.gfigure import GFigure

def sigtilde(signat, sigloc, n):
    return np.sum([signat(np.arange(n)-n0) for n0 in sigloc], axis=0)

def generate_resid(signat, snr, siglen, fs, fss, ordf, stdz=0.0, arate=0, signata=None, seed=0):
    rng = np.random.default_rng(seed)
    df = (fs/fss)/ordf
    sigloc = np.arange(0, siglen, df, dtype=float)
    sigloc += rng.standard_normal(len(sigloc))*stdz
    signal = sigtilde(signat, sigloc, siglen)

    sigloca = rng.uniform(0, siglen, arate)
    signal += sigtilde(signata, sigloca, siglen)

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

def _nmse_shift(sigest, sigtruefunc, shiftmax=100):
    siglen = len(sigest)
    sigestpad = np.pad(sigest, shiftmax)
    sigestpad /= np.linalg.norm(sigest)
    sigtrue = sigtruefunc(np.arange(siglen))
    sigtrue /= np.linalg.norm(sigtrue)
    # Since both signature estimate and true signature are normalised,
    # it is not neccesary to normalise the MSE estimate, i.e. by dividing
    # by the true signal energy.
    nmse = np.sum([(sigestpad[i:i+siglen] - sigtrue)**2 for i in range(2*shiftmax)], axis=-1)
    n = np.arange(2*shiftmax)-shiftmax
    return nmse, n

def estimate_nmse(sigest, sigtruefunc, shiftmax=100):
    nmse, n = _nmse_shift(sigest, sigtruefunc, shiftmax)
    idxmin = np.argmin(nmse)
    return nmse[idxmin], n[idxmin]


def general_snr_experiment(n_anomalous=0, mc_iterations=10):

    # synthethic data generator parameters
    siglen = 100000 # length of synthetic signal
    fs = 51200 # virtual sample frequency
    fss = 1000/60 # shaft frequency
    ordf = 5.0 # fault order
    avg_event_period = (fs/fss)/ordf # in samples
    sigfunc = lambda n: (n>=0)*np.sinc(n/8+1)
    sigfunca = lambda n: (n>=0)*np.sinc(n/2+1)

    snr_to_eval = np.logspace(-2, 0, 5)

    sigestlen = 400
    sigestshift = -150

    ordc = ordf
    ordmin = ordc-.5
    ordmax = ordc+.5

    medfiltsize = 100

    rmse = np.zeros((3, len(snr_to_eval), mc_iterations), dtype=float)

    for i, snr in enumerate(snr_to_eval):
        for j in range(mc_iterations):
            # generate residuals at given snr
            resid = generate_resid(sigfunc,
                                   snr,
                                   siglen,
                                   fs,
                                   fss,
                                   ordf,
                                   signata = sigfunca,
                                   arate = n_anomalous,
                                   seed=j)

            initial_filters = np.zeros((2,medfiltsize), dtype=float)
            # impulse
            initial_filters[0, medfiltsize//2] = 1
            initial_filters[0, medfiltsize//2+1] = -1
            # step
            initial_filters[1, :medfiltsize//2] = 1
            initial_filters[1, medfiltsize//2:] = -1

            scores = np.zeros((len(initial_filters),), dtype=float)
            medfilts = np.zeros_like(initial_filters)

            for k, initial_filter in enumerate(initial_filters):
                scores[k], medfilts[k] = algorithms.score_med(resid,
                                                            initial_filter,
                                                            ordc,
                                                            ordmin,
                                                            ordmax,)
            # IRFS method. Residuals are filtered using matched filter.
            residf = algorithms.medfilt(resid, medfilts[np.argmax(scores)])
            spos1 = algorithms.enedetloc(residf, ordmin, ordmax)
            # estimate signature using IRFS
            irfs_result = algorithms.irfs(resid, spos1, ordmin, ordmax,
                                          sigsize=sigestlen, sigshift=sigestshift)

            # estimate signature using MED and peak detection
            medout = algorithms.medfilt(resid, medfilts[0])
            medenv = abs(scipy.signal.hilbert(medout.y))
            medpeaks, _ = scipy.signal.find_peaks(medenv, distance=avg_event_period/2)
            sigest_med = estimate_signat(resid, medpeaks, sigestlen, sigestshift)
            
            # estimate signature using SK and peak detection
            skout = algorithms.skfilt(resid)
            skenv = abs(skout.y)
            skpeaks, _ = scipy.signal.find_peaks(skenv, distance=avg_event_period/2)
            sigest_sk = estimate_signat(resid, skpeaks, sigestlen, sigestshift)

            rmse[0, i, j], _ = estimate_nmse(irfs_result.sigest, sigfunc, 1000)
            rmse[1, i, j], _ = estimate_nmse(sigest_med, sigfunc, 1000)
            rmse[2, i, j], _ = estimate_nmse(sigest_sk, sigfunc, 1000)

    legend = ["IRFS", "MED", "SK"]
    styles = ["o-", "^-", "d-"]

    G = GFigure(xaxis=10*np.log10(snr_to_eval),
                yaxis=np.mean(rmse, axis=-1),
                styles=styles,
                legend=legend,
                xlabel="SNR (dB)",
                ylabel="NMSE")
    return G

class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
        """Experiment for testing RMSE estimation"""

        siglen = 100
        shift = -50
        sigfunc = lambda n: (n>=0)*np.sinc(n/8+1)
        sigest = sigfunc(np.arange(siglen)+shift)
        sigest += np.random.randn(siglen)*.1
        shiftmax = 100
        rmse, n = _nmse_shift(sigest, sigfunc, shiftmax)
        
        G = GFigure(xaxis=n, yaxis=rmse)
        return G

    def experiment_1002(l_args):
        """Benchmark methods under varying SNR for signature type A"""

        G = general_snr_experiment(n_anomalous=0)
        return G

    def experiment_1003(l_args):
        """Benchmark methods under varying SNR for signature type A"""
        
        G = general_snr_experiment(n_anomalous=100)
        return G

    def experiment_1004(l_args):
        import matplotlib
        import matplotlib.pyplot as plt
        print("Combining previous experiments")

        l_G2 = ExperimentSet.load_GFigures(1002)[0]
        l_G3 = ExperimentSet.load_GFigures(1003)[0]
        
        G = GFigure(figsize=(3.5, 2.5))
        G.l_subplots = l_G2.l_subplots + l_G3.l_subplots
        
        # edit loaded GFigure
        signatlabels = ["A", "B"]
        for i, subplt in enumerate(G.l_subplots):
            subplt.ylabel = f"NMSE ({signatlabels[i]})"
            subplt.xlabel = ""
        G.l_subplots[-1].xlabel = "SNR (dB)"

        # draw GFigure and customise plotting
        matplotlib.rcParams.update({"font.size": 8})
        fig = G.plot()
        ax = fig.get_axes()
        ax[0].legend(ncol=3, bbox_to_anchor=(0.5, 1.5), loc="upper center")
        for i in range(len(ax)):
            #ax[i].set_yscale("log")
            ax[i].grid(visible=True, which="both", axis="both")
            ax[0].get_shared_x_axes().join(ax[0], ax[i])

            if i>0: # skip first subplot
                ax[i].get_legend().remove()
            if i<len(ax)-1: # skip last subplot
                ax[i].set_xticklabels([])
        
        plt.tight_layout()
        plt.show()
    
    def experiment_1007(l_args):
        # synthethic data generator parameters
        seed = 0
        siglen = 100000 # length of synthetic signal
        fs = 51200 # virtual sample frequency
        fss = 1000/60 # shaft frequency
        ordf = 5.0 # fault order
        sigfunc = lambda n: (n>=0)*np.sinc(n/8+1)
        sigfunca = lambda n: (n>=0)*np.sinc(n/2+1)
        snr = 1
        resid = generate_resid(sigfunc,
                               snr,
                               siglen,
                               fs,
                               fss,
                               ordf,
                               signata = sigfunca,
                               arate = 100,
                               seed=seed)
        return GFigure(xaxis=resid.x, yaxis=resid.y)