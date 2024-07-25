from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

from faultevent.signal import Signal
import algorithms

from simsim import experiment, presentation


output_path = "results/synth"


# synthethic signal parameters
siglen = 100000 # length of synthetic signal
fs = 51200 # virtual sample frequency
fss = 1000/60 # shaft frequency
ordf = 5.0 # fault order
avg_event_period = (fs/fss)/ordf # in samples
sigfunc = lambda n: (n>=0)*np.sinc(n/8+1)
sigfunca = lambda n: (n>=0)*np.sinc(n/2+1)


def sigtilde(signat, sigloc, n):
    return np.sum([signat(np.arange(n)-n0) for n0 in sigloc], axis=0)


def generate_resid(signat, snr, siglen, fs, fss, ordf,
                   stdz=0.0, arate=0, signata=None, seed=0,
                   interference_std: float = 0,
                   interference_cf: float | None = None,
                   interference_bw: float | None = None):
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

    if not (interference_cf is None or interference_bw is None):
        Wn_low = interference_cf - interference_bw/2
        Wn_high = interference_cf + interference_bw/2
        interference = rng.normal(0, interference_std, resid.shape)
        sos = scipy.signal.butter(4, Wn=(Wn_low, Wn_high), btype="bandpass",
                                  output="sos", fs=fs)
        interference = scipy.signal.sosfilt(sos, interference)
        resid += interference

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


def nmse_shift(sigest, sigtruefunc, shiftmax=100):
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
    nmse, n = nmse_shift(sigest, sigtruefunc, shiftmax)
    idxmin = np.argmin(nmse)
    return nmse[idxmin], n[idxmin]



def rmse_benchmark(resid, sigfunc, avg_event_period, ordf):

    sigestlen = 400
    sigestshift = -150

    ordc = ordf
    ordmin = ordc-.5
    ordmax = ordc+.5

    medfiltsize = 100

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

    rmse_irfs, _ = estimate_nmse(irfs_result.sigest, sigfunc, 1000)
    rmse_med, _ = estimate_nmse(sigest_med, sigfunc, 1000)
    rmse_sk, _ = estimate_nmse(sigest_sk, sigfunc, 1000)

    return rmse_irfs, rmse_med, rmse_sk


def general_snr_experiment(snr, seed, n_anomalous):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""

    # generate residuals at given snr
    resid = generate_resid(sigfunc,
                           snr,
                           siglen,
                           fs,
                           fss,
                           ordf,
                           signata = sigfunca,
                           arate = n_anomalous,
                           seed=seed)

    return rmse_benchmark(resid, sigfunc, avg_event_period, ordf)


def snr_monte_carlo(n_anomalous=0, mc_iterations=10):
    """For a set of SNR-values, runs the general snr experiment multiple
    times using multiprocessing."""
    
    snr_to_eval = np.logspace(-2, 0, 5)
    list_args = []
    for snr in snr_to_eval:
        for seed in range(mc_iterations):
            args = (snr, seed, n_anomalous)
            list_args.append(args)

    with Pool() as p:
        rmse = p.starmap(general_snr_experiment, list_args)
    
    return snr_to_eval, rmse


@experiment(output_path)
def mc_snr_signature():
    """Synthetic data experiment A - No anomlaous events present"""
    return snr_monte_carlo(n_anomalous=0)


@experiment(output_path)
def mc_snr_signature_anomalous():
    """Synthetic data experiment B - Anomalous events present"""
    return snr_monte_carlo(n_anomalous=100)


def interference_experiment(cf, seed):
    """General random interference experiment. This function is called
    by monte-carlo experiments using multiprocessing and therefore needs
    to be defined on module-level."""

    # generate residuals with interference component at given central frequency
    resid = generate_resid(sigfunc,
                           1.0,
                           siglen,
                           fs,
                           fss,
                           ordf,
                           interference_std=0.05,
                           interference_cf=cf,
                           interference_bw=1000,
                           seed=seed)

    return rmse_benchmark(resid, sigfunc, avg_event_period, ordf)


@experiment(output_path)
def mc_interference():
    """For a set of central frequencies, runs the random interference
    experiment multiple times using multiprocessing."""
    
    cf_to_eval = [4e3, 8e3, 12e3, 18e3, 22e3]
    list_args = []
    for cf in cf_to_eval:
        for seed in range(10):
            args = (cf, seed)
            list_args.append(args)

    with Pool() as p:
        rmse = p.starmap(interference_experiment, list_args)
    
    return cf_to_eval, rmse


@presentation(output_path, ["mc_snr_signature", "mc_snr_signature_anomalous"])
def present_snr(results: list[npt.ArrayLike, tuple]):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ylabels = ["A", "B"]
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    for i, result in enumerate(results):
        snr_to_eval, rmse = result
        rmse = np.reshape(rmse, (len(snr_to_eval), -1, 3))
        snr = 10*np.log10(snr_to_eval)
        ax[i].plot(snr, np.mean(rmse, 1), marker=markers[i])
        ax[i].set_ylabel(f"NMSE ({ylabels[i]})")
        ax[i].grid()
        ax[i].set_yticks([0.0, 0.5, 1.0])
        ax[i].set_xticks(range(-20, 1, 5))
    
    ax[-1].set_xlabel("SNR (dB)")
    ax[0].legend(legend, ncol=len(legend), loc="upper center",
                 bbox_to_anchor=(0.5, 1.3))
    
    plt.show()


@presentation(output_path, "mc_interference")
def present_interference(result: tuple[npt.ArrayLike, tuple]):
    fig, ax = plt.subplots(1, 1, sharex=True)
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    cf_to_eval, rmse = result
    rmse = np.reshape(rmse, (len(cf_to_eval), -1, 3))
    ax.plot(cf_to_eval, np.mean(rmse, 1))
    ax.set_ylabel(f"NMSE")
    ax.grid()
    ax.set_yticks([0.0, 0.5, 1.0])

    ax.set_xlabel("Central frequency")
    ax.legend(legend, ncol=len(legend), loc="upper center")
    
    plt.show()