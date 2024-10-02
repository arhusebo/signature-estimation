from multiprocessing import Pool
from typing import TypedDict
from collections import deque

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
    sigest_ = sigest.copy()
    sigest_ /= np.linalg.norm(sigest_)
    siglen = len(sigest_)
    sigtrue = sigtruefunc(np.arange(siglen))
    sigtrue /= np.linalg.norm(sigtrue)

    if shiftmax>0:
        sigestpad = np.pad(sigest_, shiftmax)
        # Since both signature estimate and true signature are normalised,
        # it is not neccesary to normalise the MSE estimate, i.e. by dividing
        # by the true signal energy.
        nmse = np.sum([(sigestpad[i:i+siglen] - sigtrue)**2 for i in range(2*shiftmax)], axis=-1)
        n = np.arange(2*shiftmax)-shiftmax
    else:
        nmse = np.sum((sigest - sigtrue)[np.newaxis,:]**2, axis=-1)
        n = np.zeros((1,))

    return nmse, n


def estimate_nmse(sigest, sigtruefunc, shiftmax=100):
    nmse, n = nmse_shift(sigest, sigtruefunc, shiftmax)
    idxmin = np.argmin(nmse)
    return nmse[idxmin], n[idxmin]


class BenchmarkResults(TypedDict):
    method: str
    rmse: float
    sigest: npt.ArrayLike
    sigshift: int
    filter_output: Signal | None


def rmse_benchmark(resid, sigfunc, avg_event_period, ordc,
                   return_intermediate = False) -> list[BenchmarkResults]:

    sigestlen = 400
    sigestshift = -150

    ordc = ordc
    ordmin = ordc-.5
    ordmax = ordc+.5
    faults = {"":(ordmin, ordmax)}

    medfiltsize = 100

    score_med_results = algorithms.score_med(resid, medfiltsize, faults)
    residf = score_med_results["filtered"]

    # IRFS method
    spos1 = algorithms.enedetloc(residf, search_intervals=[(ordmin, ordmax)])
    irfs = algorithms.irfs(resid, spos1, ordmin, ordmax, sigestlen, sigestshift)
    irfs_result, = deque(irfs, maxlen=1)

    irfs_out = np.correlate(resid.y, irfs_result["sigest"], mode="valid")
    irfs_filt = Signal(irfs_out, resid.x[:-len(irfs_result["sigest"])+1],
                        resid.uniform_samples)


    # estimate signature using MED and peak detection
    medout = algorithms.med_filter(resid, medfiltsize, "impulse")
    medenv = abs(scipy.signal.hilbert(medout.y))
    medpeaks, _ = scipy.signal.find_peaks(medenv, distance=avg_event_period/2)
    sigest_med = estimate_signat(resid, medpeaks, sigestlen, sigestshift)
    
    # estimate signature using SK and peak detection
    skout = algorithms.skfilt(resid)
    skenv = abs(skout.y)
    skpeaks, _ = scipy.signal.find_peaks(skenv, distance=avg_event_period/2)
    sigest_sk = estimate_signat(resid, skpeaks, sigestlen, sigestshift)

    rmse_irfs, shift_irfs = estimate_nmse(irfs_result["sigest"], sigfunc, 1000)
    rmse_med, shift_med = estimate_nmse(sigest_med, sigfunc, 1000)
    rmse_sk, shift_sk = estimate_nmse(sigest_sk, sigfunc, 1000)

    results: list[BenchmarkResults] = [
        {
            "method": "IRFS",
            "rmse": rmse_irfs,
            "sigest": irfs_result["sigest"],
            "sigshift": shift_irfs,
            "filter_output": irfs_filt if return_intermediate else None,
        },
        {
            "method": "MED",
            "rmse": rmse_med,
            "sigest": sigest_med,
            "sigshift": shift_med,
            "filter_output": medout if return_intermediate else None,
        },
        {
            "method": "SK",
            "rmse": rmse_sk,
            "sigest": sigest_sk,
            "sigshift": shift_sk,
            "filter_output": skout if return_intermediate else None,
        }
    ]

    return results


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

    results = rmse_benchmark(resid, sigfunc, avg_event_period, ordf)
    return [r["rmse"] for r in results]


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


def interference_experiment(kwargs: dict):
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
                           interference_std=kwargs["interf_std"],
                           interference_cf=kwargs["interf_cfreq"],
                           interference_bw=1000,
                           seed=kwargs["seed"])

    results = rmse_benchmark(resid, sigfunc, avg_event_period, ordf)
    return [r["rmse"] for r in results]


@experiment(output_path)
def mc_interference():
    """For a set of central frequencies, runs the random interference
    experiment multiple times using multiprocessing."""
    
    interf_std = np.linspace(0.0, 1.0, 10)#[0.0, 0.05, 0.1, 0.2, 1.0]
    interf_cfreq = [8e3, 16e3]
    args = [] 
    for cfreq in interf_cfreq:
        for std in interf_std:
            for seed in range(1):
                kwargs = {
                    "interf_std": std,
                    "interf_cfreq": cfreq,
                    "seed": seed
                }
                args.append(kwargs)

    with Pool() as p:
        rmse = p.map(interference_experiment, args)
    
    return interf_std, interf_cfreq, rmse


@experiment(output_path)
def interference_signature():
    """Estimate signatures under one interference condition using each
    benchmark method."""

    resid = generate_resid(signat = sigfunc,
                           snr = 1.0,
                           siglen = siglen,
                           fs = fs,
                           fss = fss,
                           ordf = ordf,
                           interference_std=0.2,
                           interference_cf=16e3,
                           interference_bw=1000)

    
    results = rmse_benchmark(resid, sigfunc, avg_event_period, ordf,
                             return_intermediate=True)
    return results, resid


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
def present_interference(result):
    interf_std, interf_cfreq, rmse = result
    fig, ax = plt.subplots(1, len(interf_cfreq), sharex=True)
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(rmse, (len(interf_cfreq), len(interf_std), -1, 3))
    for i, cfreq in enumerate(interf_cfreq):
        ax[i].plot(interf_std, np.mean(rmse[i,:], 1))
        ax[i].set_ylabel(f"NMSE")
        ax[i].grid()
        ax[i].set_yticks([0.0, 0.5, 1.0])

        ax[i].set_xlabel("WGN STD")
        ax[i].set_title(f"Interference\ncentral frequency: {round(cfreq/1e3)} kHz")
        ax[i].legend(legend, )
        
        ax[i].invert_xaxis()
    plt.show()


@presentation(output_path, "interference_signature")
def present_interference_signature(results_):
    results, resid = results_
    fig, ax = plt.subplots(len(results)+1, 1, sharex=True)
    for i, result in enumerate(results):
        x = np.arange(len(result["sigest"]))-result["sigshift"]
        ax[i].plot(x, result["sigest"])
        ax[i].set_ylabel(result["method"])
    
    x = np.arange(-400, 400)
    sigtrue = sigfunc(x)
    ax[-1].plot(x, sigtrue)

    fig, ax = plt.subplots(len(results)+1, 1, sharex=True)
    for i, result in enumerate(results):
        ax[i].plot(result["filter_output"].x, result["filter_output"].y)
        ax[i].set_ylabel(result["method"])

    ax[-1].plot(resid.x, resid.y)
    ax[-1].set_xlabel("Revs")
    plt.show()