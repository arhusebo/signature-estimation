from multiprocessing import Pool
from typing import TypedDict, Callable, NotRequired, Any
from dataclasses import dataclass
from collections import deque
from collections.abc import Sequence
import itertools
from functools import partial

import numpy as np
import numpy.typing as npt
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from faultevent.signal import Signal
from faultevent.util import estimate_signature
from faultevent.event import event_spectrum

import algorithms
import data
from data.synth import generate_vibration, avg_fault_period,\
    VibrationDescriptor, VibrationData,\
    DEFAULT_FAULT_SIGNATURE, DEFAULT_ANOMALY_SIGNATURE
import sizeregr
import util
from config import load_config

from simsim import experiment, presentation, ExperimentStatus

cfg = load_config()

OUTPUT_PATH = "results/synth"
MC_ITERATIONS = cfg.get("mc_iterations", 30)
MAX_WORKERS = cfg.get("max_workers", None)


def estimate_signat(signal: Signal, indices: Sequence[int],
                    siglen: int, shift: int) -> Sequence[float]:
    """Estimates the signature from the signal, signal indices,
    a desired signature length and a shift."""
    n_samples = 0
    running_sum = np.zeros((siglen,))
    for idx in indices:
        if idx+shift >= 0 and idx+shift+siglen < len(signal):
            n_samples += 1
            running_sum += signal.y[idx+shift:idx+shift+siglen]
    sig = running_sum/n_samples
    return sig


@dataclass
class MethodResult:
    name: str
    sigest: npt.NDArray[np.float64]
    eosp: npt.NDArray[np.float64]


def benchmark(vibdata: VibrationData,
              irfs_params: algorithms.IRFSParams,
              fault_index: int = 0,
              sigestlen: int = 400,
              sigestshift: int = -150,
              medfiltsize: int = 100,) -> list[MethodResult]:
    """Using the given `VibrationData` benchmark IRFS against signature
    estimates obtained using
    - spectral kurtosis (SK),
    - autoregressive (AR) pre-filtering followed by SK,
    - minimum-entropy deconvolution (MED),
    - autoregressive (AR) pre-filtering followed by MED, and finally
    - a compound method using a combination of the previous algorithms.
    """

    fault = vibdata.desc["faults"][fault_index]
    avg_event_period = avg_fault_period(vibdata.desc, fault_index)

    dataname = vibdata.desc["healthy_component"]["dataname"]
    # vib = util.get_armodel(dataname).process(genres["signal"])
    vib = vibdata.signal

    armodel = util.get_armodel(dataname)
    mlmodel = util.get_mlmodel(dataname)

    resid_ar = armodel.residuals(vib)
    resid_ml = mlmodel.residuals(vib)

    #score_med_results = algorithms.score_med(resid_ml, medfiltsize, [(ordmin, ordmax)])
    #residf = score_med_results["filtered"]

    # IRFS method
    irfs = algorithms.irfs(irfs_params, resid_ml)
    for i, irfs_result in enumerate(irfs):
        if i >= 10: break

    sigest_irfs = estimate_signature(data=resid_ml, length=sigestlen, x=irfs_result.eot,
                                     weights=irfs_result.certainty)
    # irfs_out = np.correlate(resid_ml.y, irfs_result["sigest"], mode="valid")
    # irfs_filt = Signal(irfs_out, resid_ml.x[:-len(irfs_result["sigest"])+1],
    #                     resid_ml.uniform_samples)

    # estimate signature using MED and peak detection
    medout = algorithms.med_filter(vib, medfiltsize, "impulse")
    medenv = abs(scipy.signal.hilbert(medout.y))
    medpeaks, _ = scipy.signal.find_peaks(medenv, distance=avg_event_period/2)
    sigest_med = estimate_signature(data=vib, length=sigestlen, indices=medpeaks+sigestshift)
    
    # estimate signature using SK and peak detection
    skout = algorithms.skfilt(vib)
    skenv = abs(skout.y)
    skpeaks, _ = scipy.signal.find_peaks(skenv, distance=avg_event_period/2)
    sigest_sk = estimate_signature(data=vib, length=sigestlen, indices=skpeaks+sigestshift)

    # estimate signature using AR-MED and peak detection
    armedout = algorithms.med_filter(resid_ar, medfiltsize, "impulse")
    armedenv = abs(scipy.signal.hilbert(armedout.y))
    armedpeaks, _ = scipy.signal.find_peaks(armedenv, distance=avg_event_period/2)
    sigest_armed = estimate_signature(data=vib, length=sigestlen, indices=armedpeaks+sigestshift)
    
    # estimate signature using AR-SK and peak detection
    arskout = algorithms.skfilt(resid_ar)
    arskenv = abs(arskout.y)
    arskpeaks, _ = scipy.signal.find_peaks(arskenv, distance=avg_event_period/2)
    sigest_arsk = estimate_signature(data=vib, length=sigestlen, indices=arskpeaks+sigestshift)
    
    # Compound method from
    # https://www.papers.phmsociety.org/index.php/phmconf/article/download/3522/phmc_23_3522
    cmout = algorithms.skfilt(armedout)
    cmenv = abs(cmout.y)
    cmpeaks, _ = scipy.signal.find_peaks(cmenv, distance=avg_event_period/2)
    sigest_cm = estimate_signature(data=vib, length=sigestlen, indices=cmpeaks+sigestshift)

    results = [MethodResult("irfs", sigest_irfs, irfs_result.eot),
               MethodResult("med", sigest_med, medout.x[medpeaks]),
               MethodResult("sk", sigest_sk, skout.x[skpeaks]),
               MethodResult("armed", sigest_armed, armedout.x[armedpeaks]),
               MethodResult("arsk", sigest_arsk, arskout.x[arskpeaks]),
               MethodResult("cm", sigest_cm, cmout.x[cmpeaks]),]

    return results


# --- NMSE experiments --------------------------------------------------------

def nmse_shift(signature, estimate, shiftmax=100):
    # TODO: Continue here
    sigest = estimate.copy()
    sigest /= np.linalg.norm(sigest)
    siglen = len(sigest)
    
    sigtrue = signature[:siglen]
    sigtrue /= np.linalg.norm(sigtrue)

    if shiftmax>0:
        sigestpad = np.pad(sigest, shiftmax)
        # Since both signature estimate and true signature are normalised,
        # it is not neccesary to normalise the MSE estimate, i.e. by dividing
        # by the true signal energy.
        nmse = np.sum([(sigestpad[i:i+siglen] - sigtrue)**2 for i in range(2*shiftmax)], axis=-1)
        n = np.arange(2*shiftmax)-shiftmax
    else:
        nmse = np.sum((sigest - sigtrue)[np.newaxis,:]**2, axis=-1)
        n = np.zeros((1,))

    return nmse, n


def estimate_nmse(signature, maxshift, estimate):
    """Estimate the NMSE between a signature `estimate` and the true `signature`"""
    nmse, _ = nmse_shift(signature, estimate, maxshift)
    idxmin = np.argmin(nmse)
    return nmse[idxmin]


SIGNAL_ID_MAP = {
    data.DataName.UIA: "y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5",
    data.DataName.UNSW: "Test 1/6Hz/vib_000002663_06.mat",
    data.DataName.CWRU: "099",
}


def snr_experiment(seed: int, score_func: Callable[[MethodResult], Any],
                   snr: float, dataname: data.DataName,
                   anomalous: int, signature, score_args=()):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    ordf = 5.0
    fs = 51200
    seed = 0
    signature_anomalous = DEFAULT_ANOMALY_SIGNATURE(np.arange(800)).tolist()

    desc: VibrationDescriptor = {
        "length": 100000,
        "sample_frequency": fs,
        "shaft_frequency": 1000/60,
        "snr": snr,
        "healthy_component": {
            "dataname": dataname,
            "signal_id": SIGNAL_ID_MAP[dataname]
        },
        "faults": [
            {
                "ord": ordf,
                "signature": signature,
                "std": 0.01,
            }
        ],
        "anomaly": {
            "amount": anomalous,
            "signature": signature_anomalous,
        }
    }
    
    irfs_params = algorithms.IRFSParams(fmin=ordf-0.5, fmax=ordf+0.5,
                                        signature_length=200,
                                        signature_shift=-20,
                                        hyst_ed=0.8)

    vibdata = generate_vibration(desc, seed=seed)
    benchmark_results = benchmark(vibdata, irfs_params) 
    return score_func(benchmark_results, *score_args)


def benchmark_nmse(results: list[MethodResult], signature):
    return list(map(partial(estimate_nmse, signature, 1000),
               (result.sigest for result in results)))


@experiment(OUTPUT_PATH, json=True)
def ex_nmse_snr(status: ExperimentStatus):
    """Monte-carlo simulation of signature NMSE for varying SNR"""

    iterations = 1

    snr = np.logspace(-2, 0, 5).tolist()
    #dataname = [data.DataName.UIA, data.DataName.UNSW, data.DataName.CWRU]
    dataname = [data.DataName.UNSW]
    anomalous = [0, 1000]
    signature = [DEFAULT_FAULT_SIGNATURE(np.arange(800)).tolist()]

    args_list = list(itertools.product(snr, dataname, anomalous, signature))
    score_args = (signature[0],)
    
    status.max_progress = len(args_list)
    rmse = []
    for i, args in enumerate(args_list):
        args_ = [(seed, benchmark_nmse, *args, score_args) for seed in range(iterations)]
        with Pool(MAX_WORKERS) as p:
            rmse_ = p.starmap(snr_experiment, args_)
            rmse.append(np.nanmean(rmse_, axis=0).tolist())
        status.progress = i+1
        
    return args_list, rmse


# Present NMSE vs SNR results

def results_predicate(dataname: data.DataName, anomalous: bool):
    def filt(result):
        args, _ = result
        _, arg_dataname, arg_anomalous, _ = args
        if not arg_dataname == dataname:
            return False
        if bool(arg_anomalous) != anomalous:
            return False
        return True
    return filt


@presentation(ex_nmse_snr)
def pr_nmse_snr(results):
    matplotlib.rcParams.update({"font.size": 6})
    ylabels = ["A", "B"]
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    for dataname in ("uia", "unsw", "cwru"):
        _, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.0))
        plt.title(dataname.upper())
        for i, anomalous in enumerate((False, True)):
            filtres = list(filter(results_predicate(dataname, anomalous), zip(*results)))
            snr = [x[0][0] for x in filtres]
            rmse = np.array([x[-1] for x in filtres])
            snr_db = 10*np.log10(snr)
            for j in range(rmse.shape[-1]):
                ax[i].plot(snr_db, rmse[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
            ax[i].set_ylabel(f"NMSE\n{ylabels[i]}")
            ax[i].grid()
            ax[i].set_yticks([0.0, 0.5, 1.0])
            ax[i].set_xticks(range(-20, 1, 5))
            
        ax[-1].set_xlabel("SNR (dB)")
        ax[0].legend(legend, ncol=len(legend)//2, loc="upper center",
                    bbox_to_anchor=(0.5, 1.3))
    
        plt.tight_layout(pad=0.0)
    
    plt.show()


# --- Fault size experiments -------------------------------------------

def benchmark_fsize(results: list[MethodResult], fsize, model):
    # signature estimate cast to 32-bit float array for pytorch compatibility
    return list(map(partial(sizeregr.pred_err_np, fsize, model),
                   (np.array(res.sigest, dtype=np.float32) for res in results)))


@dataclass
class FseExperimentParams:
    model: sizeregr.Model
    seed: int
    snr: float
    fsize: int
    shift: int
    dataname: data.DataName


def fse(seed: int, snr: float, fsize: int):
    """FSE SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    ordf = 5.0
    fs = 51200
    signature = DEFAULT_FAULT_SIGNATURE(np.arange(800), fsize)
    model = sizeregr.Model.load(sizeregr.model_filepath())

    desc: VibrationDescriptor = {
        "length": 100000,
        "sample_frequency": fs,
        "shaft_frequency": 1000/60,
        "snr": snr,
        "healthy_component": {
            "dataname": "uia",
            "signal_id": SIGNAL_ID_MAP["uia"]
        },
        "faults": [
            {
                "ord": ordf,
                "signature": signature,
                "std": 0.01,
            }
        ]
    }
    
    irfs_params = algorithms.IRFSParams(fmin=ordf-0.5, fmax=ordf+0.5,
                                        signature_length=200,
                                        signature_shift=-40,
                                        hyst_ed=0.8)

    vibdata = generate_vibration(desc, seed=seed)
    benchmark_results = benchmark(vibdata, irfs_params)
    return benchmark_fsize(benchmark_results, fsize, model)


@experiment(OUTPUT_PATH, json=True)
def ex_fsize_snr(status: ExperimentStatus):
    """Monte-carlo simulation of fault size estimation (FSE) for varying SNR"""
    exp_params = []

    # experiment variables
    fsize = lambda i: 5+int(25*i/MC_ITERATIONS)
    snr_values = np.logspace(-2, 0, 10)

    status.max_progress = len(snr_values)

    err = []
    for progress, snr in enumerate(snr_values):
        args_ = [(i, fsize(i), snr) for i in range(MC_ITERATIONS)]
        with Pool(MAX_WORKERS) as p:
            err_ = p.starmap(fse, args_)
            err.append(np.nanmean(err_, axis=0).tolist())
        
        status.progress = progress+1

    return snr_values.tolist(), err


@presentation(ex_fsize_snr)
def pr_fsize_snr(results):
    matplotlib.rcParams.update({"font.size": 6})
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    _, ax = plt.subplots(1, 1, sharex=True, figsize=(3.5, 2.0))
    
    snr, err = results
    snr_db = 10*np.log10(snr)
    errarr = np.array(err).T
    for j, err_ in enumerate(errarr):
        ax.plot(snr_db, err_, marker=markers[j], c=cmap(cmap_idx[j]))
    ax.grid()
    ax.set_xticks(range(-20, 1, 5))
        
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Fault size (samples)")
    ax.legend(legend, ncol=len(legend)//2, loc="upper center",
              bbox_to_anchor=(0.5, 1.3))

    plt.tight_layout(pad=0.0)
    
    plt.show()

# --- EOSP experiments --------------------------------------------------------

def eosp_metric(ordf: float,
                eosp_true: Sequence[float],
                eosp_detected: Sequence[float]) -> float:
    """Compute a metric quantifying the error of detected EOSPs"""
    ptru = np.angle(event_spectrum(ordf, eosp_true))#+np.pi
    pdet = np.angle(event_spectrum(ordf, eosp_detected))#+np.pi
    pdiff = ptru - pdet
    xdiff = pdiff/(2*np.pi)/ordf
    eosp_corrected = eosp_detected - xdiff
    cdist = [np.min(abs(eosp_true-ec)) for ec in eosp_corrected]
    mcdist = np.mean(cdist)
    
    return mcdist


def common_eosp_experiment(snr, seed):
    ordf = 5.0
    desc: VibrationDescriptor = {
        "length": 100000,
        "sample_frequency": 51200,
        "shaft_frequency": 1000/60,
        "snr": snr,
        "healthy_component": {
            "dataname": "unsw",
            "signal_id": "Test 1/6Hz/vib_000002663_06.mat",
        },
        "faults": [
            {
                "ord": ordf,
                "signature": data.synth.signt_res(6.5e3, 0.001, 30, t=np.arange(800), fs=25.e3).tolist(),
                "std": 0.01,
            }
        ]
    }
    
    irfs_params = algorithms.IRFSParams(fmin=ordf-0.5, fmax=ordf+0.5,
                                        signature_length=200,
                                        signature_shift=-20,
                                        hyst_ed=0.8)

    results = benchmark(desc, irfs_params, seed=seed)
    eosp_true = results["eosp"][results["event_labels"]==1]

    metric = [eosp_metric(ordf, eosp_true, mr["eosp"])
                for mr in results["methods"].values()]

    return metric


@experiment(OUTPUT_PATH, json=False)
def ex_eosp(status: ExperimentStatus):
    snr_to_eval = np.logspace(-2, 0, 10).tolist()
    status.max_progress = len(snr_to_eval)
    for i, snr in enumerate(snr_to_eval):
        args = [(snr, seed) for seed in range(MC_ITERATIONS)]
        with Pool(MAX_WORKERS) as p:
            metric = p.starmap(common_eosp_experiment, args)
        status.progress = i+1
    
    return snr_to_eval, metric


@presentation(ex_eosp)
def pr_eosp(results):
    snr, metric = results

    metric = np.reshape(metric, (len(snr), -1, 6))
    snr = 10*np.log10(snr)
    mean_metric = np.mean(metric, 1)
    
    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    for i in range(mean_metric.shape[-1]):
        ax.plot(snr, mean_metric[:,i], marker=markers[i], c=cmap(cmap_idx[i]))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("EOSP error\n(revs)")
    ax.grid()
    ax.set_yticks([0.0, 0.02, 0.04, 0.06])
    plt.legend(legend, ncol=len(legend)//2, loc="upper center",
               bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout(pad=0.0)
    plt.show()
