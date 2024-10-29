from multiprocessing import Pool
from typing import TypedDict, Callable, NotRequired
from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.signal
import matplotlib.pyplot as plt

from faultevent.signal import Signal
import algorithms

from simsim import experiment, presentation


class FaultDescriptor(TypedDict):
    name: NotRequired[str]
    ord: float
    std: float
    signature: Callable[[int], npt.ArrayLike]


class AnomalyDescriptor(TypedDict):
    amount: int
    signature: Callable[[int], npt.ArrayLike]


class InterferenceDescriptor(TypedDict):
    sir: float
    central_frequency: float
    bandwidth: float


class ResidualDescriptor(TypedDict):
    length: int
    sample_frequency: float
    shaft_frequency: float
    faults: Sequence[FaultDescriptor]
    snr: NotRequired[float]
    anomaly: NotRequired[AnomalyDescriptor]
    interference: NotRequired[InterferenceDescriptor]


def avg_fault_period(residual: ResidualDescriptor,
                         fault_index: int = 0) -> float:
    """Computes the average number of shaft revolutions between fault event"""
    return (residual["sample_frequency"]
            /residual["shaft_frequency"])/residual["faults"][fault_index]["ord"]

output_path = "results/synth"

MC_ITERATIONS = 20

#common_fault_signature = lambda n: (n>=0)*np.sinc(n/8+1)
common_anomaly_signature = lambda n: (n>=0)*np.sinc(n/2+1)
def common_fault_signature(n):
    return (n>=0)*np.sinc(n/8+1)

common_residual: ResidualDescriptor = {
    "length": 100000,
    "sample_frequency": 51200,
    "shaft_frequency": 1000/60,
    "faults": [
        {
            "ord": 5.0,
            "signature": common_fault_signature,
            "std": 0.01,
        }
    ]
}


def sigtilde(signat, sigloc, n):
    return np.sum([signat(np.arange(n)-n0) for n0 in sigloc], axis=0)


def generate_resid(residual: ResidualDescriptor, seed=0):
    rng = np.random.default_rng(seed)
    resid = np.zeros((residual["length"]), dtype=float)

    # Generate fault signatures
    for fault in residual["faults"]:
        df = (residual["sample_frequency"]
              /residual["shaft_frequency"])/fault["ord"]
        eosp = np.arange(0, residual["length"], df, dtype=float)
        eosp += rng.standard_normal(len(eosp))*fault["std"]
        resid += sigtilde(fault["signature"], eosp, residual["length"])

    # Generate anomaly signatures
    if anomaly := residual.get("anomaly", None):
        eosp = rng.uniform(0, residual["length"], anomaly["amount"])
        resid += sigtilde(anomaly["signature"], eosp, residual["length"])
    
    # Generate signal noise
    if snr := residual.get("snr", None):
        sigpow = np.var(resid)
        noisepow = sigpow/snr
        noise = rng.standard_normal(residual["length"]) * np.sqrt(noisepow)
        resid += noise

    # Generate random interference componen
    if interference := residual.get("interference", None):
        Wn_low = interference["central_frequency"] - interference["bandwidth"]/2
        Wn_high = interference["central_frequency"] + interference["bandwidth"]/2
        interf = rng.laplace(0, 1, resid.shape)
        interf = np.sign(interf)*interf**2
        interf_sos = scipy.signal.butter(4, Wn=(Wn_low, Wn_high),
                                         btype="bandpass",
                                         output="sos",
                                         fs=residual["sample_frequency"],)
        interf = scipy.signal.sosfilt(interf_sos, interf)
        intpow = sigpow/interference["sir"]
        interf = np.sqrt(intpow)*interf/np.std(interf)
        resid += interf

    # Instantiate and return signal object
    dx = 1/(residual["sample_frequency"]/residual["shaft_frequency"])
    out = Signal.from_uniform_samples(resid, dx)
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


def rmse_benchmark(residual: ResidualDescriptor,
                   fault_index: int = 0,
                   seed: int = 0,
                   sigestlen: int = 400,
                   sigestshift: int = -150,
                   medfiltsize: int = 100,) -> Sequence[BenchmarkResults]:

    resid=generate_resid(residual, seed=seed)
    fault = residual["faults"][fault_index]
    avg_event_period = avg_fault_period(residual, fault_index)
    ordc = fault["ord"]
    sigfunc = fault["signature"]
    
    ordc = ordc
    ordmin = ordc-.5
    ordmax = ordc+.5
    faults = {"":(ordmin, ordmax)}

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

    results: Sequence[BenchmarkResults] = [
        {
            "method": "IRFS",
            "rmse": rmse_irfs,
            "sigest": irfs_result["sigest"],
            "sigshift": shift_irfs,
        },
        {
            "method": "MED",
            "rmse": rmse_med,
            "sigest": sigest_med,
            "sigshift": shift_med,
        },
        {
            "method": "SK",
            "rmse": rmse_sk,
            "sigest": sigest_sk,
            "sigshift": shift_sk,
        }
    ]

    return results


def common_snr_experiment(snr, seed, n_anomalous):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    
    residual: ResidualDescriptor = common_residual.copy()
    residual["snr"] = snr
    residual["anomaly"] = {
        "amount": n_anomalous,
        "signature": common_anomaly_signature,
    }

    results = rmse_benchmark(residual, seed=seed)
    
    return [r["rmse"] for r in results]


def snr_monte_carlo(n_anomalous=0, mc_iterations=10):
    """For a set of SNR-values, runs the general snr experiment multiple
    times using multiprocessing."""
    
    snr_to_eval = np.logspace(-2, 0, 5).tolist()
    list_args = []
    for snr in snr_to_eval:
        for seed in range(mc_iterations):
            args = (snr, seed, n_anomalous)
            list_args.append(args)

    with Pool() as p:
        rmse = p.starmap(common_snr_experiment, list_args)
    
    return snr_to_eval, rmse


@experiment(output_path, json=True)
def ex_snr():
    """Synthetic data experiment A - No anomlaous events present"""
    return snr_monte_carlo(n_anomalous=0, mc_iterations=MC_ITERATIONS)


@experiment(output_path, json=True)
def ex_snr_anomalous():
    """Synthetic data experiment B - Anomalous events present"""
    return snr_monte_carlo(n_anomalous=100, mc_iterations=MC_ITERATIONS)


def interference_experiment(interference: InterferenceDescriptor, seed=0):
    """General random interference experiment. This function is called
    by monte-carlo experiments using multiprocessing and therefore needs
    to be defined on module-level."""

    residual: ResidualDescriptor = common_residual.copy()
    residual["snr"] = 1.0
    residual["interference"] = interference

    results = rmse_benchmark(residual, seed=seed)

    return [r["rmse"] for r in results]


@experiment(output_path, json=True)
def ex_sir():
    """For a set of central frequencies, runs the random interference
    experiment multiple times using multiprocessing."""
    
    sir = np.linspace(5, 0.2, 10).tolist()
    interf_cfreq = 16e3
    args = [] 
    for sir_ in sir:
        for seed in range(MC_ITERATIONS):
            interference: InterferenceDescriptor = {
                "sir": sir_,
                "central_frequency": interf_cfreq,
                "bandwidth": 1e3,
            }
            args.append((interference, seed))

    with Pool() as p:
        rmse = p.starmap(interference_experiment, args)
    
    return sir, interf_cfreq, rmse


@experiment(output_path, json=True)
def ex_sir_frequency():
    """For a fixed signal-to-interference ratio, runs a
    "random interference" experiment for varying central frequency."""
    
    sir = 1.0
    interf_cfreq = np.linspace(2e3, 20e3, 10).tolist()
    args = [] 
    for cfreq in interf_cfreq:
        for seed in range(MC_ITERATIONS):
            interference: InterferenceDescriptor = {
                "sir": sir,
                "central_frequency": cfreq,
                "bandwidth": 1e3,
            }
            args.append((interference, seed))

    with Pool() as p:
        rmse = p.starmap(interference_experiment, args)
    
    return {
        "sir": sir,
        "interf_cfreq": interf_cfreq,
        "rmse": rmse,
    }


def diagnosis_benchmark(residual: ResidualDescriptor,
                        faults: algorithms.fault_search,
                        seed: int = 0,
                        medfiltsize=100):
    
    resid = generate_resid(residual, seed=seed)
    
    search_intervals = faults.values()
    score_med_results = algorithms.score_med(resid, medfiltsize, search_intervals)
    residf = score_med_results["filtered"]

    # IRFS method, 1st iteration
    eosp = algorithms.enedetloc(residf, search_intervals=search_intervals)
    irfs_diagnosis = algorithms.diagnose_fault(eosp, faults)
    med_diagnosis = algorithms.diagnose_fault_simple(
        signal=algorithms.med_filter(resid, medfiltsize, "impulse"),
        faults=faults,)
    sk_diagnosis = algorithms.diagnose_fault_simple(
        signal=algorithms.skfilt(resid),
        faults=faults,)
    return [
        {
            "name": "IRFS",
            "diagnosis": irfs_diagnosis,
        },
        {
            "name": "MED",
            "diagnosis": med_diagnosis,
        },
        {
            "name": "SK",
            "diagnosis": sk_diagnosis,
        }
    ]


@experiment(output_path, json=True)
def ex_diagnosis():
    """Performs a diagnosis benchmark for multiple signal realizations
    of varying levels of SNR"""
    
    faults: list[FaultDescriptor] = [
        {
            "name": "inner race",
            "ord": 5.4152,
            "signature": common_fault_signature,
            "std": 0.01,
        },
        {
            "name": "outer race",
            "ord": 3.5848,
            "signature": common_fault_signature,
            "std": 0.01
        }
    ]

    faults_to_consider: algorithms.fault_search = {
        fault["name"]: tuple(fault["ord"]+d for d in [-0.1, 0.1]) for fault in faults
    }

    snr_to_eval = np.logspace(-3, 0, 5).tolist()
    args = []
    for fault in faults:
        for snr in snr_to_eval:
            for seed in range(MC_ITERATIONS):
                residual = common_residual.copy()
                residual["faults"] = [fault]
                residual["snr"] = snr
                args.append((residual, faults_to_consider, seed))
    
    with Pool() as p:
        diagnosis_results = p.starmap(diagnosis_benchmark, args)

    # am I having a brain malfunction?
    results = [] 
    idx_results = 0
    for fault in faults:
        for snr in snr_to_eval:
            for seed in range(MC_ITERATIONS):
                results.append({
                    "fault": fault["name"],
                    "snr": snr,
                    "diagnosis": diagnosis_results[idx_results]
                })
                idx_results += 1
    return results


@presentation(ex_snr, ex_snr_anomalous)
def pr_snr(results: Sequence[npt.ArrayLike, tuple]):
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


@presentation(ex_sir)
def pr_sir(result):
    sir_to_eval, interf_cfreq, rmse = result
    plt.figure(figsize=(6.4, 3.0))
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(rmse, (len(sir_to_eval), -1, 3))

    sir = 10*np.log10(sir_to_eval)
    plt.ylabel(f"NMSE")
    plt.plot(sir, np.mean(rmse, 1))
    plt.grid()
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    plt.xlabel("SIR (dB)")
    #plt.title(f"Interference\ncentral frequency: {round(cfreq/1e3)} kHz")
    #plt.gca().invert_xaxis()
    plt.legend(legend)
    plt.tight_layout()
    plt.show()


@presentation(ex_sir_frequency)
def pr_sir_frequency(result):
    #fig, ax = plt.subplots(1, len(interf_cfreq), sharex=True, sharey=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(result["rmse"], (len(result["interf_cfreq"]), -1, 3))
    rmse = np.mean(rmse, 1)

    sir = 10*np.log10(result["sir"])
    ax.set_ylabel(f"NMSE")

    cfreq_khz = np.divide(result["interf_cfreq"], 1e3)

    ax.plot(cfreq_khz, rmse[:,0])
    ax.plot(cfreq_khz, rmse[:,1])
    ax.plot(cfreq_khz, rmse[:,2])
    ax.grid()
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax.set_xlabel("Interference central frequency (kHz)")
    ax.set_title(f"Signal-to-interference ratio: {result["sir"]} (dB)")
    
    ax.legend(legend)
    plt.tight_layout()
    plt.show()