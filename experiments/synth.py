from multiprocessing import Pool
from typing import TypedDict, Callable, NotRequired, Any, Literal
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
    mlmodel = util.get_ml2model(dataname)   # ML2: direct fault-component denoiser (feeds IRFS)

    resid_ar = armodel.residuals(vib)
    resid_ml = mlmodel.residuals(vib)

    #score_med_results = algorithms.score_med(resid_ml, medfiltsize, [(ordmin, ordmax)])
    #residf = score_med_results["filtered"]

    # IRFS method
    irfs = algorithms.irfs(irfs_params, resid_ml)
    for i, irfs_result in enumerate(irfs):
        if i >= 4: break

    # check if irfs succeeded before trying to estimate final signature
    if len(irfs_result.eoi)>0:
        sigest_irfs = estimate_signature(
            signal=vib,
            length=sigestlen,
            indices=irfs_result.eoi+sigestshift,
            weights=irfs_result.certainty)
    else:
        sigest_irfs = np.zeros((sigestlen,), dtype=float)
    # irfs_out = np.correlate(resid_ml.y, irfs_result["sigest"], mode="valid")
    # irfs_filt = Signal(irfs_out, resid_ml.x[:-len(irfs_result["sigest"])+1],
    #                     resid_ml.uniform_samples)

    # estimate signature using MED and peak detection
    medout = algorithms.med_filter(vib, medfiltsize, "impulse")
    medenv = abs(scipy.signal.hilbert(medout.y))
    medpeaks, _ = scipy.signal.find_peaks(medenv, height=np.std(medout.y)*3.0, distance=avg_event_period/2)
    sigest_med = estimate_signature(signal=vib, length=sigestlen, indices=medpeaks+sigestshift)
    
    # estimate signature using SK and peak detection
    skout = algorithms.skfilt(vib, nperseg=250)
    skenv = abs(skout.y)
    skpeaks, _ = scipy.signal.find_peaks(skenv, height=np.std(skout.y)*3.0, distance=avg_event_period/2)
    sigest_sk = estimate_signature(signal=vib, length=sigestlen, indices=skpeaks+sigestshift)

    # estimate signature using AR-MED and peak detection
    armedout = algorithms.med_filter(resid_ar, medfiltsize, "impulse")
    armedenv = abs(scipy.signal.hilbert(armedout.y))
    armedpeaks, _ = scipy.signal.find_peaks(armedenv, height=np.std(armedout.y)*3.0, distance=avg_event_period/2)
    sigest_armed = estimate_signature(signal=vib, length=sigestlen, indices=armedpeaks+sigestshift)
    
    # estimate signature using AR-SK and peak detection
    arskout = algorithms.skfilt(resid_ar, nperseg=250)
    arskenv = abs(arskout.y)
    arskpeaks, _ = scipy.signal.find_peaks(arskenv, height=np.std(arskout.y)*3.0, distance=avg_event_period/2)
    sigest_arsk = estimate_signature(signal=vib, length=sigestlen, indices=arskpeaks+sigestshift)
    
    # Compound method from
    # https://www.papers.phmsociety.org/index.php/phmconf/article/download/3522/phmc_23_3522
    cmout = algorithms.skfilt(armedout, nperseg=250)
    cmenv = abs(cmout.y)
    cmpeaks, _ = scipy.signal.find_peaks(cmenv, height=np.std(cmout.y)*3.0, distance=avg_event_period/2)
    sigest_cm = estimate_signature(signal=vib, length=sigestlen, indices=cmpeaks+sigestshift)

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


@dataclass
class SignalConfig:
    """Per-realization parameters used to build a synthetic vibration signal
    and to run/score IRFS on it."""
    desc: VibrationDescriptor
    irfs_params: algorithms.IRFSParams
    signature: npt.NDArray[np.float64]
    fsize: int
    stpres: npt.NDArray[np.float64]
    impres: npt.NDArray[np.float64]
    ordf: float


def make_signal_config(rng: np.random.Generator,
                       snr: float,
                       dataname: data.DataName,
                       anomalous: int,
                       fix_signature_params: dict = {},) -> SignalConfig:
    """Set up the parameters for one synthetic-signal realization: draw the
    fault-signature parameters (order, resonance frequency, decay, fault size)
    within the a-priori uncertainty of a fixed bearing, then build the
    fault/anomaly signatures, the `VibrationDescriptor` and the `IRFSParams`.
    Factored out of `snr_experiment` so other experiments (and the ml2 training
    synthesiser, via `data.synth.draw_signature_params`) share the same model.

    `draw_signature_params` consumes `rng` before `generate_vibration`, so the
    result is deterministic given the seed."""
    fs = 51200

    # Realized signature parameters (with slip / modal uncertainty). The
    # realized order generates the events and scores EOSPs; IRFS still searches
    # around the *nominal* order (what would be known a priori).
    params = data.synth.draw_signature_params(
            rng,
            **fix_signature_params,)
    ordf = params["ord"]
    fsize = params["d"]

    sig_f, sig_tau = params["f"], params["tau"]
    sig_fs = data.synth.SIG_FS
    sig_t = np.arange(data.synth.SIG_LEN)/sig_fs
    dt = fsize/sig_fs
    stpres = data.synth.signt_stpres(sig_f, sig_tau, sig_t)
    impres = data.synth.signt_impres(sig_f, sig_tau, sig_t)
    #signature = stpres/20 + impres
    signature = data.synth.signt_res(sig_f, sig_tau, dt, sig_t)

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

    ordf_nominal = data.synth.ORD_NOMINAL
    irfs_params = algorithms.IRFSParams(fmin=ordf_nominal-0.5, fmax=ordf_nominal+0.5,
                                        signature_length=200,
                                        signature_shift=-20,
                                        hyst_ed=0.8,
                                        hyst_mf=0.05)

    return SignalConfig(desc=desc, irfs_params=irfs_params, signature=signature,
                        fsize=fsize, stpres=stpres, impres=impres, ordf=ordf)


def snr_experiment(seed: int,
                   snr: float,
                   dataname: data.DataName,
                   anomalous: int,
                   fix_signature_params: dict = {},):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    rng = np.random.default_rng(seed)
    cfg = make_signal_config(rng, snr, dataname, anomalous, fix_signature_params)

    vibdata = generate_vibration(cfg.desc, rng=rng)
    benchmark_results = benchmark(vibdata, cfg.irfs_params)

    # nmse
    nmse = map(partial(estimate_nmse, cfg.signature, 1000),
               (res.sigest for res in benchmark_results))

    # fsize error
    fse_error = map(lambda sigest: abs(cfg.fsize-estimate_fsize(sigest, cfg.stpres, cfg.impres)),
                  (res.sigest for res in benchmark_results))

    # eosp error
    eosp_true = [eosp for (eosp, label) in zip(vibdata.eosp, vibdata.event_labels) if label==1]
    eosp_error = map(lambda r: eosp_metric(cfg.ordf, eosp_true, r.eosp), benchmark_results)

    return {
        # The nmse can be averaged across MC realizations because the energy of the denominator is 1.
        "nmse": list(nmse),
        "fse_error": list(fse_error),
        "eosp_error": list(eosp_error)
    }


def wrap_snr_experiment(kwargs):
    return snr_experiment(**kwargs)

def extract_metric(results: list[dict], name: str):
    return [r[name] for r in results]


type IndependentVarname = Literal["snr", "anomalous", "fsize", "sig_f", "sig_tau"]
type DependentVarname = Literal["nmse", "fse_error", "eosp_error"]


class ExperimentResults(TypedDict):
    nmse: list
    fse_error: list
    eosp_error: list
    indep_var: list


def ex_indep_var(indep_name: IndependentVarname, indep_var: list[Any], ex_params: dict):
    """Experiments are defined by one independent variable.
    All dependent variables are estimated for the given independent
    variable.
    This allows experiments to be run separately for different
    independent variables.
    """
    if not "fix_signature_params" in ex_params:
        ex_params["fix_signature_params"] = {}


    def ex(status: ExperimentStatus):
        status.max_progress = len(indep_var)
        results = []
        for i, x in enumerate(indep_var):
            kwargs = []
            for j in range(MC_ITERATIONS):
                entry = {"dataname": "unsw", "seed": i*MC_ITERATIONS+j, **ex_params}
                match indep_name:
                    case "snr":
                        entry["snr"] = x
                    case "anomalous":
                        entry["anomalous"] = x
                    case "ordf":
                        entry["fix_signature_params"]["ord"] = x
                    case "fsize":
                        entry["fix_signature_params"]["d"] = x
                    case "sig_f":
                        entry["fix_signature_params"]["sig_f"] = x
                    case "sig_d":
                        entry["fix_signature_params"]["sig_d"] = x
                kwargs.append(entry)
            # spawn a process for each MC iteration at the current value
            # of independent variable
            with Pool(MAX_WORKERS) as p:
                res = p.map(wrap_snr_experiment, kwargs)
                xmetric = partial(extract_metric, res)
                results.append({
                    "nmse": np.mean(xmetric("nmse"), axis=0).tolist(),
                    "fse_error": np.mean(xmetric("fse_error"), axis=0).tolist(),
                    "eosp_error": np.mean(xmetric("eosp_error"), axis=0).tolist(),
                    "indep_var": x,
                })
            status.progress = i+1
        return results

    return ex


@experiment(OUTPUT_PATH, json=True)
def ex_snr(arg):
    indep_var = np.logspace(-3, 0, 10).tolist()
    ex_params = {
            "anomalous": 0,
        }
    return ex_indep_var("snr", indep_var, ex_params)(arg)


@experiment(OUTPUT_PATH, json=True)
def ex_anomalous(arg):
    indep_var = np.arange(0, 550, 50).tolist()
    ex_params = {
            "snr": 0.03,
        }
    return ex_indep_var("anomalous", indep_var, ex_params)(arg)


@experiment(OUTPUT_PATH, json=True)
def ex_fsize(arg):
    start, stop = data.synth.FSIZE_RANGE
    step = 5
    indep_var = np.arange(start, stop, step).tolist()
    ex_params = {
            "snr": 0.01,
            "anomalous": 0,
        }
    return ex_indep_var("fsize", indep_var, ex_params)(arg)


@experiment(OUTPUT_PATH, json=True)
def ex_ordf(arg):
    indep_var = np.arange(2.0, 6.0, 0.5)
    ex_params = {
            "snr": 0.005,
            "anomalous": 0,
        }
    return ex_indep_var("fsize", indep_var, ex_params)(arg)


@experiment(OUTPUT_PATH, json=True)
def ex_sig_f(arg):
    indep_var = np.arange(*data.synth.SIG_F_RANGE, 5e2)
    ex_params = {
            "snr": 0.005,
            "anomalous": 0,
        }
    return ex_indep_var("sig_f", indep_var, ex_params)(arg)


@experiment(OUTPUT_PATH, json=True)
def ex_sig_tau(arg):
    indep_var = np.arange(*data.synth.SIG_TAU_RANGE, 0.2e-3)
    ex_params = {
            "snr": 0.005,
            "anomalous": 0,
        }
    return ex_indep_var("sig_tau", indep_var, ex_params)(arg)


# ---- Presentation ----

def results_predicate(dataname: data.DataName, anomalous: bool):
    def filt(result):
        if not result["conf"]["dataname"] == dataname:
            return False
        if bool(result["conf"]["anomalous"]) != anomalous:
            return False
        return True
    return filt


def present_experiment(indep: IndependentVarname, dep: DependentVarname,
                       results: ExperimentResults):
    """Presents the experiment of the given independent variable"""

    matplotlib.rcParams.update({"font.size": 6})
    legend = ["IRFS", "MED", "SK", "AR-MED", "AR-SK", "Compound"]
    markers = ["o", "^", "d", ".", "*", "+"]
    cmap = plt.get_cmap("tab10")
    cmap_idx = [0, 1, 2, 1, 2, 3]
    _, ax = plt.subplots(1, 1, sharex=True, figsize=(3.5, 2.0))
    
    indep_var = [r["indep_var"] for r in results]
    if indep=="snr":
        x = 10*np.log10(indep_var)
    else:
        x = indep_var

    y = np.array([r[dep] for r in results])

    match indep:
        case "snr":
            ax.set_xlabel("SNR [dB]")
        case "anomalous":
            ax.set_xlabel("Anomalous events")
        case "fsize":
            ax.set_xlabel("Fault size [Samples]")
        case "ordf":
            ax.set_xlabel("Fault order [X]")
  

    match dep:
        case "nmse":
            ax.set_ylabel("NMSE")
            ax.set_yticks([0.0, 0.5, 1.0])
        case "fse_error":
            ax.set_ylabel("Fault size error (samples)")
        case "eosp_error":
            ax.set_ylabel("EOT error\n[s]")
            #ax.set_ylim(0, 10*np.max(x[:,0]))

    # do plotting
    for i in range(y.shape[-1]):
        ax.plot(x, y[:,i], marker=markers[i], c=cmap(cmap_idx[i]))
    
    ax.grid()
    ax.legend(legend, ncol=len(legend)//2, loc="upper center",
              bbox_to_anchor=(0.5, 1.3))

    plt.tight_layout(pad=0.0)
    
    plt.show()


@presentation(ex_snr)
def pr_snr_nmse(results):
    present_experiment("snr", "nmse", results)

@presentation(ex_snr)
def pr_snr_fse(results):
    present_experiment("snr", "fse_error", results)

@presentation(ex_snr)
def pr_snr_eot(results):
    present_experiment("snr", "eosp_error", results)

@presentation(ex_anomalous)
def pr_anomalous_nmse(results):
    present_experiment("anomalous", "nmse", results)

@presentation(ex_anomalous)
def pr_anomalous_eot(results):
    present_experiment("anomalous", "eosp_error", results)

@presentation(ex_anomalous)
def pr_anomalous_fse(results):
    present_experiment("anomalous", "fse_error", results)

@presentation(ex_ordf)
def pr_ordf_nmse(results):
    present_experiment("ordf", "nmse", results)

@presentation(ex_ordf)
def pr_ordf_eot(results):
    present_experiment("ordf", "eosp_error", results)

@presentation(ex_fsize)
def pr_fsize_nmse(results):
    present_experiment("fsize", "nmse", results)

@presentation(ex_fsize)
def pr_fsize_fse(results):
    present_experiment("fsize", "fse_error", results)

@presentation(ex_sig_f)
def pr_sig_f_nmse(results):
    present_experiment("sig_f", "nmse", results)

@presentation(ex_sig_tau)
def pr_sig_tau_nmse(results):
    present_experiment("sig_tau", "nmse", results)


@presentation(ex_snr)
def pr_nmse_old(results):
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
            ax[i].plot(snr_db, rmse[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
        ax[i].set_ylabel(f"NMSE\n{ylabels[i]}")
        ax[i].grid()
        ax[i].set_yticks([0.0, 0.5, 1.0])
        #ax[i].set_xscale("log")

        ax[-1].set_xlabel("SNR [dB]")
        ax[0].legend(legend, ncol=len(legend)//2, loc="upper center")
    
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
    snr_db = 10*np.log10(snr)
    for j in range(err.shape[-1]):
        ax.plot(snr_db, err[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
    ax.grid()
        
    ax.set_xlabel("SNR [dB]")
    #ax.set_xscale("log")
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
    snr_db = 10*np.log10(snr)
    for j in range(err.shape[-1]):
        ax.plot(snr_db, err[:,j], marker=markers[j], c=cmap(cmap_idx[j]))
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("EOT error\n[s]")
    ax.grid()
    #ax.set_xscale("log")
    #ax.set_yticks([0.0, 0.02, 0.04, 0.06])
    ax.set_ylim(0, 10*np.max(err[:,0]))
    plt.legend(legend, ncol=len(legend)//2, loc="upper center",
               bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout(pad=0.0)
    plt.show()


@experiment(OUTPUT_PATH)
def ex_compare_sigest():
    

    seed = 0
    snr = 0.01
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
    
    fsize = 20
    sig_f = 6.5e3
    sig_tau = 0.001
    sig_fs = 25.e3
    sig_t = np.arange(800)
    stpres = data.synth.signt_stpres(sig_f, sig_tau, sig_t/sig_fs)
    impres = data.synth.signt_impres(sig_f, sig_tau, sig_t/sig_fs)
    signature = data.synth.signt_res(sig_f, sig_tau, fsize, sig_t, fs=sig_fs)

    fig, ax = plt.subplots(1+len(results), 1, sharex=True)
    ax[0].plot(signature)
    ax[0].set_ylabel("True")
    for i, method in enumerate(results):
        idx0 = np.argmax(np.correlate(method.sigest, stpres, mode="full"))
        idx1 = np.argmax(np.correlate(method.sigest, impres, mode="full"))
        ax[i+1].plot(method.sigest)
        ax[i+1].axvline(idx0)
        ax[i+1].axvline(idx1)
        ax[i+1].set_ylabel(method.name)

    plt.show()


# --- Fault size estimation performance ----------------------------------------

@experiment(OUTPUT_PATH, json=True)
def ex_fse2d(status: ExperimentStatus):
    """
    Apply the fault size estimator to the true signature with added WGN
    over variable SNR and fault size.
    """

    d = np.arange(20, 40)
    snr = np.logspace(-3, 2, 20)
    
    status.max_progress = len(d)*len(snr)
    
    # Signature params
    sig_f = 6.5e3
    sig_tau = 0.001
    sig_fs = 25.e3
    sig_t = np.arange(800)/sig_fs
    
    sig_entry = data.synth.signt_stpres(sig_f, sig_tau, sig_t)
    sig_exit = data.synth.signt_impres(sig_f, sig_tau, sig_t)

    def gen_signature(snr, d, rng):
        return sig + noise

    results = np.zeros((len(d), len(snr)), dtype=float)
    for i, d_ in enumerate(d):
        td = d_/sig_fs
        sig = data.synth.signt_res(sig_f, sig_tau, td, sig_t)
        for j, snr_ in enumerate(snr):
            fse = []
            for seed in range(MC_ITERATIONS):

                rng = np.random.default_rng(seed=seed)
                sigpow = np.var(sig) # just assume zero-mean
                noisepow = sigpow / snr_ # snr = sigpow / noisepow => noisepow = sigpow / snr
                noise = rng.standard_normal(len(sig_t))*np.sqrt(noisepow)

                fse.append(estimate_fsize(sig + noise, sig_entry, sig_exit))

            results[i, j] = np.mean(abs(fse-d_))
            status.progress = i*len(snr)+j+1

    return {"fsize": d.tolist(), "snr": snr.tolist(), "results": results.tolist()}


@presentation(ex_fse2d)
def pr_fse2d(results):
    img = results["results"]
    plt.imshow(results["results"])
    plt.yticks(np.arange(np.shape(img)[0]), results["fsize"])
    plt.xticks(np.arange(np.shape(img)[1]), results["snr"])
    plt.show()


# --- Signature-train recovery: AR vs ML denoiser -----------------------------

def recovery_error(estimate: npt.ArrayLike,
                   target: npt.ArrayLike,
                   max_shift: int = 300) -> tuple[float, float, int]:
    """Best-alignment error between a denoiser output `estimate` and the
    ground-truth `target` (the clean signature train).

    The estimate is slid over the target for integer offsets in
    [-max_shift, max_shift] so a method that returns a delayed/shorter signal
    (e.g. AR residuals, which drop the first `p` samples) is not penalised for
    the shift. At the best offset, two normalised errors are returned:
      - `nmse`      : scale-invariant, min over an optimal scalar gain g of
                      ||g*estimate - target||^2 / ||target||^2  ( = 1 - rho^2 )
      - `nmse_noscale`: the same with g = 1 (also penalises amplitude mismatch)
    plus the best relative offset (samples).
    """
    e = np.asarray(estimate, dtype=float)
    t = np.asarray(target, dtype=float)
    L = min(len(e), len(t)) - 2*max_shift
    if L <= 0:
        raise ValueError("signals too short for the requested max_shift")

    tt = t[max_shift:max_shift+L]                                  # fixed target window
    en_t = float(tt @ tt)
    cross = scipy.signal.correlate(e, tt, mode="valid")           # <e[k:k+L], tt>
    en_e = scipy.signal.fftconvolve(e*e, np.ones(L), mode="valid")  # <e[k:k+L], e[k:k+L]>
    n = min(len(cross), len(en_e))
    cross, en_e = cross[:n], en_e[:n]

    rho2 = cross**2 / (en_e*en_t + 1e-30)
    best = int(np.argmax(rho2))
    nmse = float(max(0.0, 1.0 - rho2[best]))
    nmse_noscale = float((en_e[best] - 2*cross[best] + en_t)/en_t)
    return nmse, nmse_noscale, best - max_shift


def ex_signature_recovery(mc: int = MC_ITERATIONS,
                          snr_db: float = -25.0,
                          dataname: data.DataName = data.DataName.UNSW,
                          fsize_interval: tuple[int, int] = (10, 40)):
    """Compare how well the AR and ML denoisers recover the clean fault
    signature train from a noisy `healthy + signature-train` signal, at a
    fixed SNR.

    For each Monte-Carlo realization a signal is built via `make_signal_config`
    + `generate_vibration(..., return_healthy=True)` (no anomalies). The true
    signature train is `signal - healthy`. Each denoiser's residual is scored
    against it with `recovery_error` (best offset), and the per-method errors
    are printed."""
    snr = 10.0**(snr_db/10.0)
    estimators = {
        "AR": util.get_armodel(dataname),
        "ML": util.get_mlmodel(dataname),
    }
    # add the direct fault-component denoiser if it has been trained
    import ml2
    if ml2.model_filepath(dataname).exists():
        estimators["ML2"] = ml2.load_model(dataname)

    errs = {name: [] for name in estimators}
    errs_noscale = {name: [] for name in estimators}

    for seed in range(mc):
        rng = np.random.default_rng(seed)
        cfg = make_signal_config(rng, snr, dataname, anomalous=0)
        vibdata, healthy = generate_vibration(cfg.desc, rng=rng, return_healthy=True)
        true_train = vibdata.signal.y - healthy.y          # clean signature train

        for name, model in estimators.items():
            resid = model.residuals(vibdata.signal)
            nmse, nmse_noscale, _ = recovery_error(resid.y, true_train)
            errs[name].append(nmse)
            errs_noscale[name].append(nmse_noscale)

    print(f"\nSignature-train recovery @ SNR = {snr_db:.0f} dB  "
          f"[{dataname}, {mc} MC realizations]")
    print("error = best-offset NMSE (lower is better)\n")
    print(f"  {'method':<6}{'NMSE (scale-inv)':>20}{'NMSE (unit gain)':>20}")
    for name in estimators:
        a = np.array(errs[name])
        b = np.array(errs_noscale[name])
        print(f"  {name:<6}{a.mean():>11.4f} +/-{a.std():<5.3f}"
              f"{b.mean():>13.4f} +/-{b.std():<5.3f}")

    return {"nmse": errs, "nmse_noscale": errs_noscale}
