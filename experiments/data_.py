from collections import deque

import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from dataclasses import dataclass

import faultevent.signal as sig
import faultevent.event as evt
import faultevent.util as utl

import algorithms

from simsim import experiment, presentation


output_path = "results/data"


def detect_and_sort(filt: sig.Signal, ordc, ordmin, ordmax, weightfunc=None, maxevents=10000):
    """Detects events using peak detection and sorts them by peak height.
    Returns 'number of detections' in ascending order and the respective
    event spectrum magnitude evaluated at the fault order. Used to score each
    method."""
    rps = (filt.x[1] - filt.x[0]) # revs per sample, assuming uniform samples
    fps = rps*ordc # fault occurences per sample
    spf = int(1/fps) # samples per fault occurence
    # in some cases, the filtered signal is already analytic:
    if np.iscomplexobj(filt.y): env = abs(filt.y)
    # otherwise, calculate the analytic signal:
    else: env = abs(scipy.signal.hilbert(filt.y))
    # detect events
    peaks, properties = scipy.signal.find_peaks(env, height=0, distance=spf/2)
    spos = filt.x[peaks]
    if weightfunc is not None: u = weightfunc(spos)
    else: u = np.ones_like(spos, dtype=int)
    uy = u*properties["peak_heights"]
    idx_sorted = np.argsort(uy)[::-1]
    # Estimate fault order. Uses all detected peaks, so may be inaccurate if
    # detections are poor. Function 'find_order' is expensive, hence order is
    # estimated once rather than for every number of peaks.
    ordf, _ = utl.find_order(spos, ordmin, ordmax)
    nvals = min(len(peaks), maxevents)
    ndets = np.arange(nvals)+1
    mags = np.zeros((nvals,), dtype=float)
    for i in range(nvals):
        spos = filt.x[peaks[idx_sorted][:i+1]]
        mags[i] = abs(evt.event_spectrum(ordf, spos))
    return ndets, mags


@dataclass
class MethodOutput:
    method_name: str
    n_detections: list[int]
    magnitudes: list[float]
    signal_filtered: sig.Signal

    def plot_scores(self, ax: plt.Axes):
        frac = self.magnitudes/self.n_detections
        ax.plot(self.n_detections, frac)
        ax.set_label(self.method_name)


@dataclass
class Output:
    data_name: str 
    signal: sig.Signal
    resid: sig.Signal
    method_outputs: list[MethodOutput]
    ordc: float
    n_events_max: int
    irfs_result: algorithms.IRFSIteration
    residf: sig.Signal



def benchmark_experiment(data_name, sigsize, sigshift, signal, resid, ordc,
                         medfiltsize, sknperseg,
                         use_irfs_eosp = False):
    """Wrapper function of a general experiment to test all benchmark methods
    on one set of data.
    
    Arguments:
    signal -- faultevent.signal.Signal object containing vibrations in shaft domain
    resid -- faultevent.signal.Signal object containing AR residuals in shaft domain
    ordc -- characteristic fault order
    medfiltsize -- MED filter size
    sknbands -- number of frequency bands for SK estimation
    
    Keyword arguments:
    use_irfs_eosp -- whether to use EOSP estimates from IRFS method or
    peak detection algorithm
    """

    ordmin = ordc-.5
    ordmax = ordc+.5

    score_med_results = algorithms.score_med(resid, medfiltsize, [(ordmin, ordmax)])
    residf = score_med_results["filtered"]

    # IRFS method.
    spos1 = algorithms.enedetloc(residf, search_intervals=[(ordmin, ordmax)])
    irfs = algorithms.irfs(resid, spos1, ordmin, ordmax, sigsize, sigshift)
    irfs_result, = deque(irfs, maxlen=1)

    if use_irfs_eosp:
        irfs_val_to_sort = irfs_result["magnitude"] * irfs_result["certainty"]
        irfs_idx = np.argsort(irfs_val_to_sort)[::-1]
        irfs_nvals = len(irfs_result["eosp"])
        irfs_ndets = np.arange(irfs_nvals)+1
        irfs_mags = np.zeros((irfs_nvals,), dtype=float)
        for i in range(irfs_nvals):
            spos = irfs_result["eosp"][irfs_idx][:i+1]
            irfs_mags[i] = abs(evt.event_spectrum(irfs_result["ordf"], spos))

    else:
        irfs_out = np.correlate(resid.y, irfs_result["sigest"], mode="valid")
        irfs_filt = sig.Signal(irfs_out, resid.x[:-len(irfs_result["sigest"])+1],
                            resid.uniform_samples)
        def irfs_weight(spos):
            z = evt.map_circle(irfs_result["ordf"], spos)
            u = scipy.stats.vonmises.pdf(z, irfs_result["kappa"], loc=irfs_result["mu"])
            return u
        irfs_ndets, irfs_mags = detect_and_sort(irfs_filt, ordc, ordmin, ordmax, weightfunc=irfs_weight)
    
    print("IRFS done.")

    # MED method. Signal is filtered using filter obtained by MED.
    med_filt = algorithms.med_filter(signal, medfiltsize, "impulse")
    med_ndets, med_mags = detect_and_sort(med_filt, ordc, ordmin, ordmax)
    print("MED done.")

    # AR-MED method. Residuals are filtered using filter obtained by AR-MED.
    armed_filt = algorithms.med_filter(resid, medfiltsize, "impulse")
    armed_ndets, armed_mags = detect_and_sort(armed_filt, ordc, ordmin, ordmax)
    print("AR-MED done.")

    # SK method. Signal is filtered using filter maximising SK.
    sk_filt = algorithms.skfilt(signal, sknperseg)
    sk_ndets, sk_mags = detect_and_sort(sk_filt, ordc, ordmin, ordmax)
    print("SK done.")

    # AR-SK method. Residuals are filtered using filter maximising SK.
    arsk_filt = algorithms.skfilt(resid, sknperseg)
    arsk_ndets, arsk_mags = detect_and_sort(arsk_filt, ordc, ordmin, ordmax)
    print("AR-SK done.")

    # Compound method from
    # https://www.papers.phmsociety.org/index.php/phmconf/article/download/3522/phmc_23_3522
    cm_filt = algorithms.skfilt(armed_filt, sknperseg)
    cm_ndets, cm_mags = detect_and_sort(cm_filt, ordc, ordmin, ordmax)
    print("Compound method done.")

    n_events_max = ordc*signal.x[-1]

    irfs_output = MethodOutput("IRFS", irfs_ndets, irfs_mags, irfs_filt)
    med_output = MethodOutput("MED", med_ndets, med_mags, med_filt)
    armed_output = MethodOutput("AR-MED", armed_ndets, armed_mags, armed_filt)
    sk_output = MethodOutput("SK", sk_ndets, sk_mags, sk_filt)
    arsk_output = MethodOutput("AR-SK", arsk_ndets, arsk_mags, arsk_filt)
    cm_output = MethodOutput("Compound", cm_ndets, cm_mags, cm_filt)

    method_outputs = [irfs_output, med_output, armed_output, sk_output, arsk_output, cm_output]

    results = Output(
        data_name, signal, resid, method_outputs, ordc, n_events_max, irfs_result, residf)

    return results


@experiment(output_path)
def ex_uia():
    from data.uia import UiADataLoader
    from data import uia_path
    dl = UiADataLoader(uia_path)
    mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
    mf = dl["y2016-m09-d24/00-40-22 1000rpm - 51200Hz - 100LOR.h5"]
    
    rpm = 1000 # angular speed in rpm
    fs = 51200 # sample frequency
    signalt = mf.vib # signal in time domain
    model = sig.ARModel.from_signal(mh.vib[:10000], 117) # AR model
    residt = model.residuals(signalt) # AR residuals in time domain

    # Angular speed of these measurements are approximately constant,
    # no resampling is applied.
    signal = sig.Signal.from_uniform_samples(signalt.y, (rpm/60)/fs)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return benchmark_experiment("UIA",
                                sigsize = 400,
                                sigshift = -150,
                                signal = signal,
                                resid = resid,
                                ordc = 6.7087166, # contact angle corrected
                                medfiltsize = 100,
                                sknperseg = 1000,)


@experiment(output_path)
def ex_unsw():
    from data.unsw import UNSWDataLoader
    from data import unsw_path
    dl = UNSWDataLoader(unsw_path)
    mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
    mf = dl["Test 1/6Hz/vib_000356575_06.mat"]
    
    angfhz = 6 # angular frequency in Hz
    fs = 51200 # sample frequency
    signalt = mf.vib # signal in time domain
    model = sig.ARModel.from_signal(mh.vib[:10000], 41) # AR model
    residt = model.residuals(signalt) # AR residuals in time domain

    # Angular speed of these measurements are approximately constant,
    # no resampling is applied.
    signal = sig.Signal.from_uniform_samples(signalt.y, angfhz/fs)
    resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

    return benchmark_experiment("UNSW",
                                sigsize = 200,
                                sigshift = -100,
                                signal = signal,
                                resid = resid,
                                ordc = 3.56,
                                medfiltsize = 100,
                                sknperseg = 256,)


@experiment(output_path)
def ex_cwru():
    from data.cwru import CWRUDataLoader
    from data import cwru_path
    dl = CWRUDataLoader(cwru_path)
    mh = dl["100"]
    mf = dl["175"]
    
    rpm = dl.signal_info("175")["rpm"] # angular frequency in Hz
    fs = 48e3 # sample frequency
    signalt = mf.vib # signal in time domain
    model = sig.ARModel.from_signal(mh.vib[:10000], 75) # AR model
    residt = model.residuals(signalt) # AR residuals in time domain

    # Angular speed of these measurements are approximately constant,
    # no resampling is applied.
    signal = sig.Signal.from_uniform_samples(signalt.y, (rpm/60)/fs)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return benchmark_experiment("CWRU",
                                sigsize = 400,
                                sigshift = -150,
                                signal = signal,
                                resid = resid,
                                ordc = 5.4152,
                                medfiltsize = 100,
                                sknperseg = 256,)


def _present_benchmark_general(ax: plt.Axes, results: Output):
    for method_output in results.method_outputs:
        frac = method_output.magnitudes/method_output.n_detections
        ax.plot(method_output.n_detections, frac, label=method_output.method_name)
    ax.axvline(results.n_events_max, label="Max events", ls="--", c="k")


@presentation(ex_uia, ex_unsw, ex_cwru)
def present_benchmarks(all_results: list[Output]):
    fig, ax = plt.subplots(nrows=len(all_results))
    for i, results in enumerate(all_results):
        _present_benchmark_general(ax[i], results)
        ax[i].set_ylabel(f"True positive rate\n({results.data_name})")
        ax[i].grid(which="both")
    ax[-1].set_xlabel("Detections")
    ax[0].legend(ncol=3, bbox_to_anchor=(0.5, 1.6), loc="upper center")
    plt.show()


@presentation(ex_uia)
def present_intermediate_uia(results: Output):
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(results.signal.x, results.signal.y)
    ax[0].set_ylabel("Signal")
    ax[1].plot(results.resid.x, results.resid.y)
    ax[1].set_ylabel("Residual")
    ax[2].plot(results.residf.x, results.residf.y)
    ax[2].set_ylabel("Pre-filtered")
    ax[3].plot(results.method_outputs[0].signal_filtered.x, results.method_outputs[0].signal_filtered.y)
    ax[3].set_ylabel("IRFS-fitlered")
    ax[3].set_xlabel("Revs")
    ax[3].set_xlim(1, 12)
    plt.show()


@presentation(ex_uia, ex_unsw, ex_cwru)
def present_event_spectrum(all_results: list[Output]):
    ord = np.arange(0, 10, 0.01)
    fig, ax = plt.subplots(nrows=len(all_results))
    for i, results in enumerate(all_results):
        spos = results.irfs_result["eosp"]
        ordf = results.irfs_result["ordf"]
        evsp = evt.event_spectrum(ord, spos)
        ax[i].plot(ord, abs(evsp)) 
        ax[i].axvline(ordf, ls="--", c="k", label="Fault order")
        ax[i].axhline(len(spos), ls="-", c="gray", label="#Detected events")
        ax[i].set_xlim(ord[0], ord[-1])
        ax[i].set_ylabel(f"Event spectrum\nmag. ({results.data_name})")
        ax[i].set_xlabel("Order [X]")
    ax[0].legend(ncol=2, bbox_to_anchor=(0.5, 1.4), loc="upper center")
    plt.show()


@presentation(ex_uia, ex_unsw, ex_cwru)
def present_periodic_transform(all_results: list[Output]):
    fig, ax = plt.subplots(nrows=len(all_results), sharex=True)
    zpdf = np.linspace(0, 2*np.pi, 1000)
    for i, results in enumerate(all_results):
        ordf = results.irfs_result["ordf"]
        spos = results.irfs_result["eosp"]
        mu = results.irfs_result["mu"]
        kappa = results.irfs_result["kappa"]
        z = evt.map_circle(ordf, spos)
        n = evt.period_number(ordf, spos)
        pdf = scipy.stats.vonmises.pdf(zpdf, kappa, loc=mu)
        ax_pdf = ax[i].twinx()
        ax[i].scatter(z, n, label="Transformed EOSPs", c="k")
        ax_pdf.plot(zpdf, pdf, label="Certainty metric")
        ax_pdf.set_ylabel("Certainty metric")
        ax[i].set_ylabel("#Period")
        ax[i].set_xticks([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi],
                         [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
        ax[i].set_xlim(0, 2*np.pi)
    
    ax[-1].set_xlabel("Offset")
    ax[0].legend(bbox_to_anchor=(0.5, 1.4), loc="upper center")
    plt.show()
