from multiprocessing import Pool
from typing import TypedDict, Callable, NotRequired
from collections import deque
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from faultevent.signal import Signal
from faultevent.event import event_spectrum
import algorithms

from simsim import experiment, presentation



OUTPUT_PATH = "results/synth"
MC_ITERATIONS = 30



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
    snir: NotRequired[float]
    anomaly: NotRequired[AnomalyDescriptor]
    interference: NotRequired[InterferenceDescriptor]


def avg_fault_period(residual: ResidualDescriptor,
                         fault_index: int = 0) -> float:
    """Computes the average number of shaft revolutions between fault event"""
    return (residual["sample_frequency"]
            /residual["shaft_frequency"])/residual["faults"][fault_index]["ord"]


#common_fault_signature = lambda n: (n>=0)*np.sinc(n/8+1)
DEFAULT_ANOMALY_SIGNATURE = lambda n: (n>=0)*np.sinc(n/2+1)
def DEFAULT_FAULT_SIGNATURE(n):
    return (n>=0)*np.sinc(n/8+1)

COMMON_RESIDUAL: ResidualDescriptor = {
    "length": 100000,
    "sample_frequency": 51200,
    "shaft_frequency": 1000/60,
    "faults": [
        {
            "ord": 5.0,
            "signature": DEFAULT_FAULT_SIGNATURE,
            "std": 0.01,
        }
    ]
}


def sigtilde(signat: Callable[[int], float], sigloc: Sequence[int], n: int):
    """Samples a train of signatures given signature function 'signat'
    evaluated at 0, 1, ..., 'n' with signature offsets 'sigloc'."""
    return np.sum([signat(np.arange(n)-n0) for n0 in sigloc], axis=0)


def generate_resid(residual: ResidualDescriptor, seed=0):
    """Generates a residual signal according to the provided
    ResidualDescriptor. Always generates the same result unless a
    different value of 'seed' is used.
    
    Returns a dictionary containing the signal, event
    labels and EOSPs.
    """
    rng = np.random.default_rng(seed)
    resid = np.zeros((residual["length"]), dtype=float)
    eosp_ = []
    elbl_ = []

    # Generate fault signatures
    for i, fault in enumerate(residual["faults"]):
        df = (residual["sample_frequency"]
              /residual["shaft_frequency"])/fault["ord"]
        eosp = np.arange(0, residual["length"], df, dtype=float)
        eosp += rng.standard_normal(len(eosp))*fault["std"]
        resid += sigtilde(fault["signature"], eosp, residual["length"])

        eosp_.append(eosp)
        elbl_.append(np.ones_like(eosp, dtype=int)+i)

    # Generate anomaly signatures
    if anomaly := residual.get("anomaly", None):
        eosp = rng.uniform(0, residual["length"], anomaly["amount"])
        resid += sigtilde(anomaly["signature"], eosp, residual["length"])

        eosp_.append(eosp)
        elbl_.append(np.zeros_like(eosp, dtype=int))
    
    
    sigpow = np.var(resid) # total signatures power

    # Generate signal noise
    if snr := residual.get("snr", None):
        noisepow = sigpow/snr
        noise = rng.standard_normal(residual["length"]) * np.sqrt(noisepow)
    else:
        noise = np.zeros_like(resid)

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
        interf *= np.sqrt(intpow)/np.std(interf)
    else:
        interf = np.zeros_like(resid)

    ni = noise + interf # noise and interference
    if (snir := residual.get("snir", None)) and (snr or interference):
        # SNIR = var(resid) / var(noise + interf)
        # var(noise + interf) = var(resid) / SNIR
        # std(noise + interf) = sqrt(var(resid) / SNIR)
        # std(noise + interf) = std(resid) / sqrt(SNIR)
        pow_ni = sigpow/snir
        ni *= np.sqrt(pow_ni)/np.std(ni)
    
    resid += ni

    # Instantiate and return signal object
    dx = 1/(residual["sample_frequency"]/residual["shaft_frequency"])
    out = Signal.from_uniform_samples(resid, dx)
    event_shaft_positions = np.concatenate(eosp_)*dx
    event_labels = np.concatenate(elbl_)
    return {
        "signal": out,
        "eosp": event_shaft_positions,
        "event_labels": event_labels,
    }


def estimate_signat(resid: Signal, indices: Sequence[int],
                    siglen: int, shift: int) -> Sequence[float]:
    """Estimates the signature from the residual signal, signal indices,
    a desired signature length and a shift."""
    n_samples = 0
    running_sum = np.zeros((siglen,))
    for idx in indices:
        if idx+shift >= 0 and idx+shift+siglen < len(resid):
            n_samples += 1
            running_sum += resid.y[idx+shift:idx+shift+siglen]
    sig = running_sum/n_samples
    return sig


class MethodResults(TypedDict):
    sigest: Sequence[float]
    eosp: Sequence[float]


class BenchmarkResults(TypedDict):
    methods: dict[str, MethodResults]
    eosp: Sequence[float]
    event_labels: Sequence[int]


def benchmark(residual: ResidualDescriptor,
              fault_index: int = 0,
              seed: int = 0,
              sigestlen: int = 400,
              sigestshift: int = -150,
              medfiltsize: int = 100,) -> BenchmarkResults:

    genres=generate_resid(residual, seed=seed)
    resid = genres["signal"]
    fault = residual["faults"][fault_index]
    avg_event_period = avg_fault_period(residual, fault_index)
    ordc = fault["ord"]
    
    ordc = ordc
    ordmin = ordc-.5
    ordmax = ordc+.5

    score_med_results = algorithms.score_med(resid, medfiltsize, [(ordmin, ordmax)])
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

    results: BenchmarkResults = {
        "eosp": genres["eosp"],
        "event_labels": genres["event_labels"],
        "methods": {
            "irfs": {"sigest": irfs_result["sigest"], "eosp": irfs_result["eosp"]},
            "med": {"sigest": sigest_med, "eosp": medout.x[medpeaks]},
            "sk": {"sigest": sigest_sk, "eosp": skout.x[skpeaks]},
        },
    }

    return results


# --- NMSE experiments --------------------------------------------------------

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


def common_snr_experiment(snr, seed, n_anomalous):
    """General SNR experiment. This function is called by monte-carlo
    experiments using multiprocessing and therefore needs to be defined
    on module-level."""
    
    residual: ResidualDescriptor = COMMON_RESIDUAL.copy()
    residual["snr"] = snr
    residual["anomaly"] = {
        "amount": n_anomalous,
        "signature": DEFAULT_ANOMALY_SIGNATURE,
    }
    sigfunc = residual["faults"][0]["signature"]
    results = benchmark(residual, seed=seed)
    nmse = [estimate_nmse(mr["sigest"], sigfunc, 1000)[0]
            for mr in results["methods"].values()]
    
    return nmse


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


@experiment(OUTPUT_PATH, json=True)
def ex_snr():
    """Synthetic data experiment A - No anomlaous events present"""
    return snr_monte_carlo(n_anomalous=0, mc_iterations=MC_ITERATIONS)


@experiment(OUTPUT_PATH, json=True)
def ex_snr_anomalous():
    """Synthetic data experiment B - Anomalous events present"""
    return snr_monte_carlo(n_anomalous=100, mc_iterations=MC_ITERATIONS)


@presentation(ex_snr, ex_snr_anomalous)
def pr_snr(results: Sequence[npt.ArrayLike, tuple]):
    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3.5, 2.0))
    ylabels = ["A", "B"]
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    for i, result in enumerate(results):
        snr_to_eval, rmse = result
        rmse = np.reshape(rmse, (len(snr_to_eval), -1, 3))
        snr = 10*np.log10(snr_to_eval)
        mean_rmse = np.mean(rmse, 1)
        for j in range(mean_rmse.shape[-1]):
            ax[i].plot(snr, mean_rmse[:,j], marker=markers[j])
        ax[i].set_ylabel(f"NMSE\n{ylabels[i]}")
        ax[i].grid()
        ax[i].set_yticks([0.0, 0.5, 1.0])
        ax[i].set_xticks(range(-20, 1, 5))
    
    ax[-1].set_xlabel("SNR (dB)")
    ax[0].legend(legend, ncol=len(legend), loc="upper center",
                 bbox_to_anchor=(0.5, 1.3))
    
    plt.tight_layout(pad=0.0)
    plt.show()



def interference_experiment(interference: InterferenceDescriptor, seed=0):
    """General random interference experiment. This function is called
    by monte-carlo experiments using multiprocessing and therefore needs
    to be defined on module-level."""

    residual: ResidualDescriptor = COMMON_RESIDUAL.copy()
    residual["snr"] = 1.0
    residual["interference"] = interference

    results = benchmark(residual, seed=seed)
    sigfunc = residual["faults"][0]["signature"]
    nmse = [estimate_nmse(mr["sigest"], sigfunc, 1000)[0]
            for mr in results["methods"].values()]

    return nmse


@experiment(OUTPUT_PATH, json=True)
def ex_sir():
    """For a set of central frequencies, runs the random interference
    experiment multiple times using multiprocessing."""
    
    # sir = np.linspace(5, 0.2, 10).tolist()
    sir_max = 5.0
    sir_min = 0.2
    sir = np.logspace(np.log10(sir_max), np.log10(sir_min), 5).tolist()
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


@experiment(OUTPUT_PATH, json=True)
def ex_sir_frequency():
    """For a fixed signal-to-interference ratio, runs a
    "random interference" experiment for varying central frequency."""
    
    sir = 1.0
    interf_cfreq = np.linspace(1e3, 22e3, 10).tolist()
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


@presentation(ex_sir)
def pr_sir_basic(result):
    
    plt.figure(figsize=(3.5, 1.0))
    matplotlib.rcParams.update({"font.size": 6})

    
    # plot NMSE against SIR
    sir_to_eval, interf_cfreq, rmse = result
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(rmse, (len(sir_to_eval), -1, 3))

    sir = 10*np.log10(sir_to_eval)
    plt.ylabel(f"NMSE")
    mean_rmse = np.mean(rmse, 1)
    for i in range(mean_rmse.shape[-1]):
        plt.plot(sir, mean_rmse[:,i], marker=markers[i])
    plt.grid()
    # plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.yticks([0.0, 0.5, 1.0])
    # plt.ylim(0, 1.25)
    plt.xticks(np.round(sir, 2))

    plt.xlabel("SIR (dB)")
    # plt.title(f"Interference\ncentral frequency: {round(cfreq/1e3)} kHz")
    #plt.gca().invert_xaxis()

    #h, l = ax[0].get_legend_handles_labels()
    plt.legend(legend)
    # plt.legend(legend, ncol=len(legend), loc="upper center",)
                # bbox_to_anchor=(0.5, 1.5))
    plt.tight_layout(pad=0.0)
    plt.show()


@presentation(ex_sir, ex_sir_frequency)
def pr_sir(result_):
    result = result_[1]
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(6.4, 2.5))

    # plot NMSE against interference central frequency
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(result["rmse"], (len(result["interf_cfreq"]), -1, 3))
    rmse = np.mean(rmse, 1)

    sir = 10*np.log10(result["sir"])
    ax[0].set_ylabel(f"NMSE")

    cfreq_khz = np.divide(result["interf_cfreq"], 1e3)

    for i in range(rmse.shape[1]):
        ax[0].plot(cfreq_khz, rmse[:,i])
    ax[0].grid()
    ax[0].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax[0].set_xlabel("Interference central frequency (kHz)")
    # ax[0].set_title(f"Signal-to-interference ratio: {sir} (dB)")
    
    # plot NMSE against SIR
    result = result_[0]
    sir_to_eval, interf_cfreq, rmse = result
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    rmse = np.reshape(rmse, (len(sir_to_eval), -1, 3))

    sir = 10*np.log10(sir_to_eval)
    # ax[1].set_ylabel(f"NMSE")
    ax[1].plot(sir, np.mean(rmse, 1))
    ax[1].grid()
    ax[1].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

    ax[1].set_xlabel("SIR (dB)")
    # plt.title(f"Interference\ncentral frequency: {round(cfreq/1e3)} kHz")
    #plt.gca().invert_xaxis()

    #h, l = ax[0].get_legend_handles_labels()
    plt.figlegend(legend)
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
    ax.set_title(f"Signal-to-interference ratio: {sir} (dB)")
    
    ax.legend(legend)
    plt.tight_layout()
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
    residual: ResidualDescriptor = COMMON_RESIDUAL.copy()
    residual["snr"] = snr
    results = benchmark(residual, seed=seed)
    eosp_true = results["eosp"][results["event_labels"]==1]
    ordf = residual["faults"][0]["ord"]

    metric = [eosp_metric(ordf, eosp_true, mr["eosp"])
                for mr in results["methods"].values()]

    return metric


@experiment(OUTPUT_PATH, json=False)
def ex_eosp():
    snr_to_eval = np.logspace(-2, 0, 5).tolist()
    list_args = []
    for snr in snr_to_eval:
        for seed in range(MC_ITERATIONS):
            args = (snr, seed)
            list_args.append(args)

    with Pool(4) as p:
        metric = p.starmap(common_eosp_experiment, list_args)
    
    return snr_to_eval, metric


@presentation(ex_eosp)
def pr_eosp(results):
    snr, metric = results

    metric = np.reshape(metric, (len(snr), -1, 3))
    snr = 10*np.log10(snr)
    mean_metric = np.mean(metric, 1)
    
    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(figsize=(3.5, 1.0))
    legend = ["IRFS", "MED", "SK"]
    markers = ["o", "^", "d"]
    for i in range(mean_metric.shape[-1]):
        ax.plot(snr, mean_metric[:,i], marker=markers[i])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("EOSP error\n(revs)")
    ax.grid()
    ax.legend(legend)
    # ax.legend(legend, ncol=len(legend), loc="upper center",)
            #   bbox_to_anchor=(0.5, 1.3))
    plt.tight_layout(pad=0.0)
    plt.show()



# --- Diagnosis experiments (unused) ------------------------------------------

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


@experiment(OUTPUT_PATH, json=True)
def ex_diagnosis():
    """Performs a diagnosis benchmark for multiple signal realizations
    of varying levels of signal-to-noise-and-interference ratio (SNIR)"""
    
    faults: list[FaultDescriptor] = [
        {
            "name": "inner race",
            "ord": 5.4152,
            "signature": DEFAULT_FAULT_SIGNATURE,
            "std": 0.01,
        },
        {
            "name": "outer race",
            "ord": 3.5848,
            "signature": DEFAULT_FAULT_SIGNATURE,
            "std": 0.01
        }
    ]

    faults_to_consider: algorithms.fault_search = {
        fault["name"]: tuple(fault["ord"]+d for d in [-0.1, 0.1]) for fault in faults
    }

    snir_to_eval = np.logspace(-2, 0, 10).tolist()
    args = []
    for fault in faults:
        for snir in snir_to_eval:
            for seed in range(MC_ITERATIONS):
                residual = COMMON_RESIDUAL.copy()
                residual["snr"] = 1.0
                residual["interference"] = {
                    "sir": 5.0,
                    "bandwidth": 1e3,
                    "central_frequency": 16e3,
                }
                residual["faults"] = [fault]
                residual["snir"] = snir
                args.append((residual, faults_to_consider, seed))
    
    with Pool() as p:
        diagnosis_results = p.starmap(diagnosis_benchmark, args)

    # am I having a brain malfunction?
    results = [] 
    idx_results = 0
    for fault in faults:
        for snir in snir_to_eval:
            for seed in range(MC_ITERATIONS):
                results.append({
                    "fault": fault["name"],
                    "snir": snir,
                    "order": fault["ord"],
                    "methods": diagnosis_results[idx_results]
                })
                idx_results += 1
    return results


@presentation(ex_diagnosis)
def pr_diagnosis(results):
    faults = ["inner race", "outer race"]
    methods = ["IRFS", "MED", "SK"]

    fig, ax = plt.subplots(len(faults), 1, sharex=True)
    for i, fault in enumerate(faults):
        fault_results = [fault_result for fault_result in results
                         if fault_result["fault"] == fault]

        snir_list = sorted(list(set([result["snir"] for result in fault_results])))
        rate = []
        for snir in snir_list:
            rate.append([])
            for j, method in enumerate(methods):
                methods_gen = (
                    result["methods"][j] for result in fault_results
                    if result["snir"] == snir)
                method_rate = 0 
                for count, method in enumerate(methods_gen):
                    correct = method["diagnosis"]["fault"] == fault
                    method_rate += int(correct)
                rate[-1].append(100*method_rate/(count+1))


        snir_list_db = 10*np.log10(snir_list)
        ax[i].plot(snir_list_db, rate, label=methods)
        ax[i].legend()
        ax[i].set_ylabel(f"{fault.capitalize()} accuracy (%)")
        
    ax[-1].set_xlabel("SNIR (db)")
    
    plt.show()