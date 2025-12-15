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


SIGNAL_ID_MAP = {
    data.DataName.UIA: "y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5",
    data.DataName.UNSW: "Test 1/6Hz/vib_000002663_06.mat",
    data.DataName.CWRU: "099",
}


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

    # check if irfs succeeded before trying to estimate final signature
    if len(irfs_result.eot)>0:
        sigest_irfs = estimate_signature(
            data=vib,
            length=sigestlen,
            indices=vib.idx_closest(irfs_result.eot)+sigestshift,
            weights=irfs_result.certainty)
    else:
        sigest_irfs = np.zeros((sigestlen,), dtype=float)
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

def nmse_shift(signature, maxshift, estimate):
    # TODO: Continue here
    sigest = estimate.copy()
    sigest /= np.linalg.norm(sigest)
    siglen = len(sigest)
    
    sigtrue = signature[:siglen]
    sigtrue /= np.linalg.norm(sigtrue)

    if maxshift>0:
        sigestpad = np.pad(sigest, maxshift)
        # Since both signature estimate and true signature are normalised,
        # it is not neccesary to normalise the MSE estimate, i.e. by dividing
        # by the true signal energy.
        nmse = np.sum([(sigestpad[i:i+siglen] - sigtrue)**2 for i in range(2*maxshift)], axis=-1)
        n = np.arange(2*maxshift)-maxshift
    else:
        nmse = np.sum((sigest - sigtrue)[np.newaxis,:]**2, axis=-1)
        n = np.zeros((1,))

    return nmse, n


def estimate_nmse(signature, maxshift, estimate):
    """Estimate the NMSE between a signature `estimate` and the true `signature`"""
    nmse, _ = nmse_shift(signature, maxshift, estimate)
    idxmin = np.argmin(nmse)
    return nmse[idxmin]


def estimate_fsize(sigest, stpres, impres):
    idx0 = np.argmax(np.correlate(sigest, stpres, mode="full"))
    idx1 = np.argmax(np.correlate(sigest, impres, mode="full"))
    return idx1-idx0


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
    mcdist = np.nanmean(cdist)
    return mcdist


def snr_experiment(seed: int,
                   snr: float,
                   dataname: data.DataName,
                   anomalous: int,
                   fsize: int):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    ordf = 5.0
    fs = 51200
    
    sig_f = 6.5e3
    sig_tau = 0.001
    sig_fs = 25.e3
    sig_t = np.arange(800)
    stpres = data.synth.signt_stpres(sig_f, sig_tau, sig_t/sig_fs)
    impres = data.synth.signt_impres(sig_f, sig_tau, sig_t/sig_fs)
    signature = data.synth.signt_res(sig_f, sig_tau, fsize, sig_t, fs=sig_fs)
    
    signature_anomalous = DEFAULT_ANOMALY_SIGNATURE(np.arange(800)).tolist()

    desc: VibrationDescriptor = {
        "length": 100000,
        "sample_frequency": fs,
        "shaft_frequency": 1000/60,
        "healthy_component": {
            "dataname": dataname,
            "signal_id": SIGNAL_ID_MAP[dataname]
        },
        "faults": [
            {
                "ord": ordf,
                "signature": signature,
                "std": 0.01,
                "snr": snr,
            }
        ],
        "anomaly": {
            "amount": anomalous,
            "signature": signature_anomalous,
            "snr": 5*snr,
        }
    }
    
    irfs_params = algorithms.IRFSParams(fmin=ordf-0.5, fmax=ordf+0.5,
                                        signature_length=200,
                                        signature_shift=-20,
                                        hyst_ed=0.8)

    vibdata = generate_vibration(desc, seed=seed)
    benchmark_results = benchmark(vibdata, irfs_params) 

    # nmse
    nmse = map(partial(estimate_nmse, signature, 1000),
               (res.sigest for res in benchmark_results))
    
    # fsize error
    fse_error = map(lambda sigest: abs(fsize-estimate_fsize(sigest, stpres, impres)),
                  (res.sigest for res in benchmark_results))
    
    # eosp error
    eosp_true = [eosp for (eosp, label) in zip(vibdata.eosp, vibdata.event_labels) if label==1]
    eosp_error = map(lambda r: eosp_metric(ordf, eosp_true, r.eosp), benchmark_results)

    return {
        "nmse": list(nmse),
        "fse_error": list(fse_error),
        "eosp_error": list(eosp_error)
    }


def wrap_snr_experiment(kwargs):
    return snr_experiment(**kwargs)

def extract_metric(results: list[dict], name: str):
    return [r[name] for r in results]


@experiment(OUTPUT_PATH, json=True)
def ex_snr(status: ExperimentStatus):
    """Monte-carlo simulation of signature NMSE for varying SNR"""

    conf = {
        "snr": np.logspace(-3, 0, 10).tolist(),
        "dataname": [data.DataName.UNSW,],
        "anomalous": [0, 10,],
        "fsize": [20,],
    }

    conf_list = list(dict(zip(conf.keys(), x)) for x in itertools.product(*conf.values()))

    status.max_progress = len(conf_list)
    results = []
    for i, conf in enumerate(conf_list):
        kwargs = ({**conf, "seed": i} for i in range(MC_ITERATIONS))
        with Pool(MAX_WORKERS) as p:
            res = p.map(wrap_snr_experiment, kwargs)
            xmetric = partial(extract_metric, res)
            results.append({
                "nmse": np.mean(xmetric("nmse"), axis=0).tolist(),
                "fse_error": np.mean(xmetric("fse_error"), axis=0).tolist(),
                "eosp_error": np.mean(xmetric("eosp_error"), axis=0).tolist(),
                "conf": conf,
            })
        status.progress = i+1
        
    return results


# Present NMSE vs SNR results

def results_predicate(dataname: data.DataName, anomalous: bool):
    def filt(result):
        if not result["conf"]["dataname"] == dataname:
            return False
        if bool(result["conf"]["anomalous"]) != anomalous:
            return False
        return True
    return filt


@presentation(ex_snr)
def pr_nmse(results):
    matplotlib.rcParams.update({"font.size": 6})
    ylabels = ["A", "B"]
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    dataname = "unsw"
    _, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.0))
    for i, anomalous in enumerate((False, True)):
        filtres = list(filter(results_predicate(dataname, anomalous), results))
        snr = [x["conf"]["snr"] for x in filtres]
        rmse = np.array([x["nmse"] for x in filtres])
        snr_db = 10*np.log10(snr)
        for j in range(rmse.shape[-1]):
            ax[i].plot(snr, rmse[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
        ax[i].set_ylabel(f"NMSE\n{ylabels[i]}")
        ax[i].grid()
        ax[i].set_yticks([0.0, 0.5, 1.0])
        ax[i].set_xscale("log")

        ax[-1].set_xlabel("SNR")
        ax[0].legend(legend, ncol=len(legend)//2, loc="upper center",
                    bbox_to_anchor=(0.5, 1.3))
    
        plt.tight_layout(pad=0.0)
    
    plt.show()

@presentation(ex_snr)
def pr_fsize(results):
    matplotlib.rcParams.update({"font.size": 6})
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    _, ax = plt.subplots(1, 1, sharex=True, figsize=(3.5, 2.0))
    
    filtres = list(filter(results_predicate("unsw", True), results))
    snr = [r["conf"]["snr"] for r in filtres]
    err = np.array([r["fse_error"] for r in filtres])
    for j in range(err.shape[-1]):
        ax.plot(snr, err[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
    ax.grid()
        
    ax.set_xlabel("SNR")
    ax.set_xscale("log")
    ax.set_ylabel("Fault size error (samples)")
    ax.legend(legend, ncol=len(legend)//2, loc="upper center",
              bbox_to_anchor=(0.5, 1.3))

    plt.tight_layout(pad=0.0)
    
    plt.show()


@presentation(ex_snr)
def pr_eosp(results):
    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    
    filtres = list(filter(results_predicate("unsw", False), results))
    snr = [r["conf"]["snr"] for r in filtres]
    err = np.array([r["eosp_error"] for r in filtres])
    for j in range(err.shape[-1]):
        ax.plot(snr, err[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
    ax.set_xlabel("SNR")
    ax.set_ylabel("EOT error\n(s)")
    ax.grid()
    ax.set_xscale("log")
    #ax.set_yticks([0.0, 0.02, 0.04, 0.06])
    plt.legend(legend, ncol=len(legend)//2, loc="upper center",
               bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout(pad=0.0)
    plt.show()


@experiment(OUTPUT_PATH)
def ex_compare_sigest():
    

    seed = 0
    snr = 0.005
    dataname = "unsw"
    anomalous = 10
    fsize = 20


    ordf = 5.0
    fs = 51200
    
    sig_f = 6.5e3
    sig_tau = 0.001
    sig_fs = 25.e3
    sig_t = np.arange(800)
    stpres = data.synth.signt_stpres(sig_f, sig_tau, sig_t/sig_fs)
    impres = data.synth.signt_impres(sig_f, sig_tau, sig_t/sig_fs)
    signature = data.synth.signt_res(sig_f, sig_tau, fsize, sig_t, fs=sig_fs)
    
    signature_anomalous = DEFAULT_ANOMALY_SIGNATURE(np.arange(800)).tolist()

    desc: VibrationDescriptor = {
        "length": 100000,
        "sample_frequency": fs,
        "shaft_frequency": 1000/60,
        "healthy_component": {
            "dataname": dataname,
            "signal_id": SIGNAL_ID_MAP[dataname]
        },
        "faults": [
            {
                "ord": ordf,
                "signature": signature,
                "std": 0.01,
                "snr": snr,
            }
        ],
        "anomaly": {
            "amount": anomalous,
            "signature": signature_anomalous,
            "snr": 5*snr,
        }
    }
    
    irfs_params = algorithms.IRFSParams(fmin=ordf-0.5, fmax=ordf+0.5,
                                        signature_length=200,
                                        signature_shift=-20,
                                        hyst_ed=0.8)

    vibdata = generate_vibration(desc, seed=seed)
    benchmark_results = benchmark(vibdata, irfs_params)

    return benchmark_results


@presentation(ex_compare_sigest)
def pr_compare_sigest(results: list[MethodResult]):

    fig, ax = plt.subplots(len(results), 1, sharex=True)
    for i, method in enumerate(results):
        ax[i].plot(method.sigest)
        ax[i].set_ylabel(method.name)
    
    plt.show()
