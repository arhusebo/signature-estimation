import numpy as np
import scipy.signal
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
from typing import TypedDict, Sequence
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import dataclass

import faultevent.signal as sig
import faultevent.event as evt
import faultevent.util as utl

import algorithms
import data
import util
from config import load_config

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


class Benchmark(TypedDict):
    signal_id: str 
    method_outputs: list[MethodOutput]
    ordc: float
    events_max: int
    irfs_result: algorithms.IRFSIteration


@dataclass
class BenchmarkParams:
    rpm: float
    siglen: int
    sigshift: int
    ordc: float
    med_filtsize: int
    sk_nperseg: int
    siglen_vib: int | None = None


def benchmark_experiment(dataname: data.DataName, signal_id: str,
                         params: BenchmarkParams) -> Benchmark:
    """A general experiment to test all benchmark methods on a signal.
    
    Arguments:
    dataname -- name of the dataset
    signal_id -- ID of the signal in the datasets dataloader
    params -- benchmark parameters defined in `BenchmarkParams`
    """

    ordmin = params.ordc-.5
    ordmax = params.ordc+.5

    dl = data.dataloader(dataname)

    armodel = util.get_armodel(dataname)
    mlmodel = util.get_mlmodel(dataname)

    signalt = dl[signal_id].vib
    signal = sig.Signal.from_uniform_samples(signalt.y, (params.rpm/60)/signalt.fs)

    resid_ar = armodel.residuals(signal)
    # resid_ml = resid_ar # for verifying method still works like before
    resid_ml = mlmodel.residuals(signal)

    # score_med_results = algorithms.score_med(resid_ml, params.med_filtsize, [(ordmin, ordmax)])
    # residf = score_med_results["filtered"]

    # IRFS method.
    spos1 = algorithms.enedetloc(resid_ml, search_intervals=[(ordmin, ordmax)])
    irfs = algorithms.irfs(resid_ml, spos1, ordmin, ordmax, params.siglen, params.sigshift,
                           vibration=signal, vibration_sigsize=params.siglen_vib)
    irfs_result, = deque(irfs, maxlen=1)
    # TODO: See if we can use the EOSPs output by `algorithms.irfs` directly
    # in `detect_and_sort` instead of what we do in the following lines
    irfs_out = np.correlate(resid_ar.y, irfs_result["sigest"], mode="valid")
    irfs_filt = sig.Signal(irfs_out, resid_ar.x[:-len(irfs_result["sigest"])+1],
                        resid_ar.uniform_samples)
    def irfs_weight(spos):
        z = evt.map_circle(irfs_result["ordf"], spos)
        u = scipy.stats.vonmises.pdf(z, irfs_result["kappa"], loc=irfs_result["mu"])
        return u
    
    irfs_ndets, irfs_mags = detect_and_sort(irfs_filt, params.ordc, ordmin, ordmax, weightfunc=irfs_weight)
    
    # MED method. Signal is filtered using filter obtained by MED.
    med_filt = algorithms.med_filter(signal, params.med_filtsize, "impulse")
    med_ndets, med_mags = detect_and_sort(med_filt, params.ordc, ordmin, ordmax)

    # AR-MED method. Residuals are filtered using filter obtained by AR-MED.
    armed_filt = algorithms.med_filter(resid_ar, params.med_filtsize, "impulse")
    armed_ndets, armed_mags = detect_and_sort(armed_filt, params.ordc, ordmin, ordmax)

    # SK method. Signal is filtered using filter maximising SK.
    sk_filt = algorithms.skfilt(signal, params.sk_nperseg)
    sk_ndets, sk_mags = detect_and_sort(sk_filt, params.ordc, ordmin, ordmax)

    # AR-SK method. Residuals are filtered using filter maximising SK.
    arsk_filt = algorithms.skfilt(resid_ar, params.sk_nperseg)
    arsk_ndets, arsk_mags = detect_and_sort(arsk_filt, params.ordc, ordmin, ordmax)

    # Compound method from
    # https://www.papers.phmsociety.org/index.php/phmconf/article/download/3522/phmc_23_3522
    cm_filt = algorithms.skfilt(armed_filt, params.sk_nperseg)
    cm_ndets, cm_mags = detect_and_sort(cm_filt, params.ordc, ordmin, ordmax)

    events_max = params.ordc*signal.x[-1]

    results: Benchmark = {
        "signal_id": signal_id,
        "ordc": params.ordc,
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


def bmexp(args):
    """With `ProcessPoolExecutor`, functions must be called using its
    `map` method, meaning arguments must be provided as a single
    object. To preserve the original signature of
    `benchmark_experiment`, this function may be used with
    `ProcessPoolExecutor.map` and the arguments provided as a tuple."""
    dataname, signal_id, params = args
    return benchmark_experiment(dataname, signal_id, params)



@experiment(output_path)
def uia() -> list[Benchmark]:
    cfg = load_config()
    data_path = data.data_path(data.DataName.UIA)
    benchmark_params = BenchmarkParams(1000, 400, -150, 6.7087166, 100, 1000, 600)
    ex_args = []
    for file_path in data_path.glob("y2016-m09-d24/*.h5"):
        if not "1000rpm" in str(file_path):
            continue # we select only the 1000rpm signals
        ex_args.append((data.DataName.UIA, file_path, benchmark_params))

    with ProcessPoolExecutor(cfg.get("max_workers", None)) as executor:
        benchmarks = list(executor.map(bmexp, ex_args))
    
    return benchmarks


@experiment(output_path)
def unsw() -> list[Benchmark]:
    cfg = load_config()
    data_path = data.data_path(data.DataName.UIA)
    benchmark_params = BenchmarkParams(360, 200, -100, 3.56, 100, 256, 600)
    ex_args =  []
    for file_path in data_path.glob("Test 1/6Hz/*.mat"):
        ex_args.append((data.DataName.UNSW, file_path, benchmark_params))
    with ProcessPoolExecutor(cfg.get("max_workers", None)) as executor:
        benchmarks = list(executor.map(bmexp, ex_args))
    return benchmarks


@experiment(output_path)
def cwru() -> list[Benchmark]:
    from data.cwru import Diagnostics, fault
    cfg = load_config()
    dl = data.dataloader(data.DataName.CWRU)
    ex_args = []
    for dl_entry in dl.info["data"]:
        match fault(dl_entry["name"])[0]:
            # case Diagnostics.OUTER:
            #     ordc = 3.5848
            case Diagnostics.INNER:
                ordc = 5.4152
            case _:
                continue

        benchmark_params = BenchmarkParams(dl_entry["rpm"], 400, -150, ordc,
                                           100, 256, 600)
        
        ex_args.append((data.DataName.CWRU, dl_entry["id"], benchmark_params))
    
    with ProcessPoolExecutor(cfg.get("max_workers", None)) as executor:
        benchmarks = list(executor.map(bmexp, ex_args))

    return benchmarks


def select_benchmarks(benchmarks: list[Benchmark],
                      include_idx: list[int] | None = None,
                      include_names: list[str] | None = None,):
    # remove failed benchmarks
    list_benchmarks = list(filter(lambda r: r is not None, benchmarks))
    selected = []
    if include_names:
        selected += list(filter(lambda r: r["data_name"] in include_names, list_benchmarks))
    if include_idx:
        selected += np.take(list_benchmarks, include_idx).tolist()
    return selected

def present_benchmarks(list_benchmarks: list[Benchmark], n: int | None = None,
                       include_idx: list[int] | None = None,
                       include_names: list[str] | None = None, show_names=False,
                       dx=1.0):
    list_benchmarks = list(filter(lambda r: r is not None, list_benchmarks))
    if include_names:
        results_to_show = list(filter(
            lambda r: r["signal_id"] in include_names, list_benchmarks))
    elif include_idx:
        results_to_show = np.take(list_benchmarks, include_idx)
    else:
        results_to_show = list_benchmarks
    
    if n:
        n = min(len(results_to_show), n)
    else:
        n = len(results_to_show)

    nrows = len(results_to_show)

    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(nrows=nrows, ncols=2, sharey=False, sharex='col',
                           gridspec_kw={"width_ratios":[3, 2]},
                           figsize=(3.5, 2.5))
    for i, results in enumerate(results_to_show):

        for method_output in results["method_outputs"]:
            frac = method_output["magnitudes"]/method_output["detections"]
            ax[i][0].plot(method_output["detections"], frac, label=method_output["name"])
        ax[i][0].axvline(results["events_max"], label="Max events", ls="--", c="k")
        if i == nrows-1:
            ax[i][0].set_xlabel("Detections")
        
        ax[i][0].set_ylabel("True\npositive rate"
                            +(f"\n{results["data_name"]}" if show_names else ""))
        # ax[i][1].yaxis.set_label_position("right")

        ax[i, 0].grid(which="both")

        sigest = results["irfs_result"]["sigest_vib"]
        x = np.arange(len(sigest))*dx
        ax[i, 1].plot(x, sigest, c="k", lw=0.5)
        ax[i][1].set_ylabel("Signature\nestimate")
        ax[i][1].set_yticks([])
        ax[-1][1].set_xlabel("Time [s]")
        # ax[0][1].set_xticks([])

    h, l = ax[0][0].get_legend_handles_labels()
    plt.figlegend(h, l, ncols=4, loc="upper center")
    plt.tight_layout(rect=(0, 0, 1, 0.85))#pad=.5, h_pad=1.5)
    plt.show()


@presentation(uia)
def present_uia(results: list[Benchmark]):
    present_benchmarks(results, include_idx=[2, 3, 4], dx=1/51200)


@presentation(unsw)
def present_unsw(results: list[Benchmark]):
    present_benchmarks(results, include_idx=[11, 12, 13], dx=1/51200)


@presentation(cwru)
def present_cwru(results: list[Benchmark]):
    present_benchmarks(results, include_names=["175", "176", "215"], dx=1/48000)


@presentation(uia, unsw, cwru)
def present_all_labeled(list_results: list[list[Benchmark]]):
    for results in list_results:
        present_benchmarks(results, show_names=True)


def present_stacked_signatures(benchmark: Benchmark,
                               dataname: data.DataName,
                               idx_eosp: Sequence[int],
                               rpm: float,
                               fs: float):
    dl = data.dataloader(dataname)
    vibt = dl[benchmark["signal_id"]].vib
    model = util.get_mlmodel(dataname)
    rest = model.residuals(vibt)
    vib = sig.Signal(vibt.y, vibt.x*rpm/60, uniform_samples=vibt.uniform_samples)
    eosp = benchmark["irfs_result"]["eosp"]
    dx = 1/fs
    sigest = benchmark["irfs_result"]["sigest_vib"]
    siglen = len(sigest)
    x = np.arange(siglen)*dx#-res.x[0]

    eosp_to_plot = eosp[idx_eosp]
    idx_vib = vib.idx_closest(eosp_to_plot)#+ar_model.p

    crt = benchmark["irfs_result"]["certainty"]
    crtsort = np.sort(crt)[::-1]
    # sigest_vib = utl.estimate_signature(vib, siglen, x=eosp, weights=crtsort,
    #                                     n0=ar_model.p)

    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(len(eosp_to_plot)+1, 1, sharex=True, sharey=False,
                           figsize=(3.5, 2.5),
                           gridspec_kw={"height_ratios": [1.0]*len(eosp_to_plot)+[2.0]})
    for i, eosp_ in enumerate(eosp_to_plot):
        # vibwin = vib[idx_vib[i]+ar_model.p:idx_vib[i]+ar_model.p+siglen]
        vibwin = vib[idx_vib[i]:idx_vib[i]+siglen]
        ax[i].plot(x, vibwin.y, c="k", lw=0.5)
        ax[i].set_ylabel("Vib.\nwindow")
        # ax[i].set_ylim(-0.1, 0.1)
        ax[i].set_yticks([])
    
    ax[-1].plot(x, sigest, lw=0.8, c="k")
    ax[-1].set_ylabel("Signature\nestimate")
    ax[-1].set_xlabel("Time [s]")
    # ax[-1].set_ylim(-0.05, 0.05)
    ax[-1].set_yticks([])

    plt.tight_layout()
    plt.show()


@presentation(uia)
def present_uia_stacked(results: list[Benchmark]):
    benchmark = select_benchmarks(results, include_idx=[2])[0]
    present_stacked_signatures(benchmark=benchmark, dataname=data.DataName.UIA,
                               idx_eosp=range(4, 8), rpm=1000, fs=51200)


@presentation(unsw)
def present_unsw_stacked(results: list[Benchmark]):
    benchmark = select_benchmarks(results, include_idx=[11])[0]
    present_stacked_signatures(benchmark=benchmark, dataname=data.DataName.UNSW,
                               idx_eosp=range(4), rpm=360, fs=51200)