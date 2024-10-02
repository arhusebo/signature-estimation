import numpy as np
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from typing import TypedDict
from concurrent.futures import ProcessPoolExecutor
from collections import deque

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


class MethodOutput(TypedDict):
    name: str
    detections: list[int]
    magnitudes: list[float]


class Output(TypedDict):
    data_name: str 
    method_outputs: list[MethodOutput]
    ordc: float
    events_max: int
    irfs_result: algorithms.IRFSResult



def benchmark_experiment(data_name, sigsize, sigshift, signal, resid, ordc,
                         medfiltsize, sknperseg):
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
    faults = {"":(ordmin, ordmax)}

    score_med_results = algorithms.score_med(resid, medfiltsize, faults)
    residf = score_med_results["filtered"]

    # IRFS method.
    spos1 = algorithms.enedetloc(residf, search_intervals=[(ordmin, ordmax)])
    irfs = algorithms.irfs(resid, spos1, ordmin, ordmax, sigsize, sigshift)
    irfs_result, = deque(irfs, maxlen=1)

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

    events_max = ordc*signal.x[-1]

    results: Output = {
        "data_name": data_name,
        "ordc": ordc,
        "events_max": events_max,
        "irfs_result": irfs_result,
        "method_outputs": [
            {
                "name": "IRFS",
                "detections": irfs_ndets,
                "magnitudes":irfs_mags,
            },
            {
                "name": "MED",
                "detections": med_ndets,
                "magnitudes": med_mags,
            },
            {
                "name": "AR-MED",
                "detections": armed_ndets,
                "magnitudes": armed_mags,
            },
            {
                "name": "SK",
                "detections": sk_ndets,
                "magnitudes": sk_mags
            },
            {
                "name": "AR-SK",
                "detections": arsk_ndets,
                "magnitudes": arsk_mags
            },
            {
                "name": "Compound",
                "detections": cm_ndets,
                "magnitudes": cm_mags
            },
        ],
    }

    return results



def ex_uia(process_kwargs):
    file_path = process_kwargs["file_path"]
    dl = process_kwargs["dl"]
    model = process_kwargs["model"]
    print("Working on file "+str(file_path))
    rpm = 1000 # angular speed in rpm
    fs = 51200 # sample frequency
    mf = dl[file_path]
    signalt = mf.vib # signal in time domain
    residt = model.residuals(signalt) # AR residuals in time domain

    # Angular speed of these measurements are approximately constant,
    # no resampling is applied.
    signal = sig.Signal.from_uniform_samples(signalt.y, (rpm/60)/fs)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)
    kwargs = {
        "data_name": "UIA",
        "sigsize": 400,
        "sigshift": -150,
        "signal": signal,
        "resid": resid,
        "ordc": 6.7087166,
        "medfiltsize": 100,
        "sknperseg": 1000,
    }
    return benchmark_experiment(**kwargs)


@experiment(output_path)
def uia():
    from data.uia import UiADataLoader
    from data import uia_path
    dl = UiADataLoader(uia_path)

    mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
    model = sig.ARModel.from_signal(mh.vib[:10000], 117) # AR model

    ex_args = []
    for i, file_path in enumerate(dl.path.glob("y2016-m09-d24/*.h5")):
        if not "1000rpm" in str(file_path):
            continue
        ex_args.append({
            "file_path": file_path,
            "model": model,
            "dl": dl,
        })

    with ProcessPoolExecutor() as executor:
        return list(executor.map(ex_uia, ex_args))


def _present_benchmark_general(ax: plt.Axes, results: Output):
    for method_output in results["method_outputs"]:
        frac = method_output["magnitudes"]/method_output["detections"]
        ax.plot(method_output["detections"], frac, label=method_output["name"])
    ax.axvline(results["events_max"], label="Max events", ls="--", c="k")


@presentation(output_path, "uia")
def present_uia(list_results):
    for results in list_results:
        plt.figure()
        ax = plt.gca()
        _present_benchmark_general(ax, results)
        plt.show()
        