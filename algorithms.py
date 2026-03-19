from collections.abc import Sequence, Generator
from typing import Literal, TypedDict
from dataclasses import dataclass, replace
import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.stats
import faultevent.event as evt
import faultevent.signal as sig
from faultevent.signal import Detector, MatchedFilterMaximumDetector
from faultevent import util as utl


@dataclass
class IRFSParams:
    fmin: float
    fmax: float
    signature_length: int
    signature_shift: int
    max_shift_error: int = 10
    threshold_trials: int = 10
    ed_window: int = 50
    hyst_ed: float = 0.8
    hyst_mf: float = 0.9


@dataclass
class IRFSIteration:
    sigest: Sequence[float]
    eoi: Sequence[int]
    eot: Sequence[float]
    certainty: Sequence[float]
    freq: float
    mu: float
    kappa: float
    threshold: float | None


def irfs(params: IRFSParams,
         signal: sig.Signal,
         ) -> Generator[IRFSIteration, None, None]:
    """Given an initial iteration, return a generator that yields
    subsequent iterations of IRFS"""

    # energy detector to estimate initial set of EOTs
    eoi0 = enedetloc(data=signal,
                     search_intervals=[(params.fmin, params.fmax)],
                     enedetsize=params.ed_window,
                     hysteresis=params.hyst_ed)
    
    det0 = sig.EnergyDetector(params.ed_window)
    stat0 = det0.statistic(signal)
    thr0, _ = utl.best_threshold(data=stat0,
                                 search_intervals=[(params.fmin, params.fmax)],
                                 hysteresis=params.hyst_ed,
                                 dettype="ed",
                                 n=params.threshold_trials)
    cmp0 = sig.Comparison.from_comparator(stat0, thr0, thr0*params.hyst_ed)

    eoi0 = sig.energy_detector_location_estimates(cmp0) + params.signature_shift
    eot0 = stat0.x[eoi0]

    if len(eoi0)==0:
        raise ValueError("energy detector did not detect any events")

    freq0, _ = evt.find_order(eot0, params.fmin, params.fmax)
    mu0, kappa0 = evt.fit_vonmises(freq0, eot0)
    z0 = evt.map_circle(freq0, eot0)
    crt0 = scipy.stats.vonmises.pdf(z0, kappa0, loc=mu0)

    sigest0 = utl.scm(signal=signal.y,
                     length=params.signature_length,
                     maxerror=params.max_shift_error,
                     eoi=eoi0,
                     weights=crt0,)
    # TODO: Correct the EOIs?

    yield IRFSIteration(sigest=sigest0,
                        eoi=eoi0,
                        eot=eot0,
                        certainty=crt0,
                        freq=freq0,
                        mu=mu0,
                        kappa=kappa0,
                        threshold=thr0,)

    normthr = None

    # subsequent iterations
    sigest = sigest0
    while True:
        det = MatchedFilterMaximumDetector(sigest)
        stat = det.statistic(signal)

        if normthr is None:
            thr, _ = utl.best_threshold(stat, [(params.fmin, params.fmax)],
                                        n=params.threshold_trials,
                                        hysteresis=params.hyst_mf,
                                        thresholds=np.linspace(np.min(stat.y), np.max(stat.y), params.threshold_trials))
            normthr = thr/np.linalg.norm(sigest)
        else:
            thr = normthr*np.linalg.norm(sigest)

        cmp = sig.Comparison.from_comparator(stat, thr, hysteresis=thr*params.hyst_mf)
        eoi = sig.matched_filter_location_estimates(cmp)+params.signature_length # TODO: Why we need to add signature length?
        if len(eoi)==0:
            break
        #eot = cmp.data.x[eoi]
        eot = stat.x[eoi]
        freq, _ = evt.find_order(eot, params.fmin, params.fmax)
        mu, kappa = evt.fit_vonmises(freq, eot)
        z = evt.map_circle(freq, eot)
        crt = scipy.stats.vonmises.pdf(z, kappa, loc=mu)
        sigest = utl.estimate_signature(
                signal=signal,
                length=params.signature_length,
                indices=eoi,
                weights=crt,)
        
        yield IRFSIteration(
            sigest=sigest,
            eoi=eoi,
            eot=eot,
            certainty=crt,
            freq=freq,
            mu=mu,
            kappa=kappa,
            threshold=thr,
        )



class DiagnosedFault(TypedDict):
    fault: str
    ord: float

#type fault_search = dict[str, tuple[float, float]]

def diagnose_fault(eosp: npt.ArrayLike, faults) -> DiagnosedFault:
    """Diagnose the fault type given event shaft positions and a list
    of dictionaries containing fault characteristics. Fault
    dictionaries must be on the form
    {
        "OR" : (4.9, 5.1),
        "IR" : (3.7, 3.8)
    }
    Assumes faulty condition.
    """

    best_score = 0.0
    for fault, interval in faults.items():
        ord, magnitude = evt.find_order(eosp, *interval, density=1e3)
        score = magnitude/np.sqrt(len(eosp)) if len(eosp) else 0.0
        if score > best_score:
            best_score = score
            best_result: DiagnosedFault = {"fault": fault, "ord": ord}
    
    return best_result


def diagnose_fault_simple(signal: sig.Signal, faults) -> DiagnosedFault:
    """Similar as diagnose_fault, but is based on the envelope spectrum."""
    best_score = 0.0
    envelope = scipy.signal.hilbert(signal.y)
    spec = np.fft.rfft(abs(envelope))
    orders = np.fft.rfftfreq(len(signal), signal.dx)
    for fault, (ord_low, ord_high) in faults.items():
        search_window = np.argwhere(np.logical_and(ord_low <= orders, orders <= ord_high))
        spec_window = spec[search_window]
        orders_window = orders[search_window]
        idx_max = np.argmax(abs(spec_window))
        score = abs(spec_window[idx_max])
        ordf = orders_window[idx_max][0]
        if score > best_score:
            best_score = score
            best_result: DiagnosedFault = {"fault": fault, "ord": ordf}
    
    return best_result


def med_initial(filter_length: int, shape: Literal["step", "impulse"]):
    initial_filter = np.zeros((filter_length,), dtype=float)
    if shape == "step":
        initial_filter[:filter_length//2] = 1
        initial_filter[filter_length//2:] = -1
    elif shape == "impulse":
        initial_filter[filter_length//2] = 1
        initial_filter[filter_length//2+1] = -1
    return initial_filter


def med_filter(signal: sig.Signal,
               filter_length: int,
               initial_type: Literal["step", "impulse"]) -> sig.Signal:
    """Applies an MED-filter to the signal"""

    initial_filter = med_initial(filter_length, initial_type)
    med_filter = medest(signal.y, initial_filter)
    return medfilt(signal, med_filter)


def medest(x: npt.ArrayLike, f0: npt.ArrayLike, its: int=10) -> np.ndarray:
    """Minimum entropy deconvolution (R. A. Wiggins, 1978).
    Given input x, estimates the filter (FIR) that maximises the output
    kurtosis (and effectively impulsiveness), where f0 is the initial
    guess."""
    L = len(f0)
    N = len(x)
    X = np.zeros((L, N))
    X[0] = x
    for l in range(1, L):
        X[l][l:] = x[:-l]
    
    f = f0
    for i in range(its):
        y = X.T@f
        fa = np.sum(y**2)/np.sum(y**4)
        fb = np.linalg.solve(X@X.T, X@(y**3).T)
        f = fa*fb
    return f


def medfilt(signal: sig.Signal, filt: npt.ArrayLike):
    """Returns a signal filtered using an MED filter"""
    n = len(filt)
    out = np.convolve(signal.y, filt, mode="valid") # filtering
    return sig.Signal(out, signal.x[n-1:],
                      uniform_samples=signal.uniform_samples)


def skfilt(signal: sig.Signal, nperseg: int = 1000):
    """Filters the given signal using a spectral kurtosis (SK) derived
    filter through an STFT estimator. The filterbank filter bandwidth
    can be controlled through the nperseg parameter. For best
    performance, this parameter should follow the two conditions given
    in the original paper by J. Antoni et al."""
    if not signal.uniform_samples:
        raise ValueError("Samples must be uniformly spaced.")
    ts = signal.x[1]-signal.x[0]
    _, _, Y = scipy.signal.stft(signal.y, nperseg=nperseg, fs=1/ts)
    S = lambda n: np.mean(abs(Y)**(2*n), axis=-1)
    K = S(2)/S(1)**2 - 2
    #K = scipy.stats.kurtosis(Y, axis=-1)
    sqrtK = np.sqrt(K*(K>0))
    Ym = Y*sqrtK[:,np.newaxis]
    t, out = scipy.signal.istft(Ym, nperseg=nperseg, fs=1/ts)
    return sig.Signal(out, t, uniform_samples=True)


def enedetloc(data: sig.Signal,
              search_intervals: list[tuple[float, float]] | None = None,
              enedetsize: int = 50,
              hysteresis: float = .8,
              threshold: float | None = None,
              threshold_trials: int = 10) -> np.ndarray:
    """Detect and return locations of events using an energy detector"""
    det = sig.EnergyDetector(enedetsize)
    stat = det.statistic(data)
    if not search_intervals is None:
        threshold, _ = utl.best_threshold(stat,
                                          search_intervals,
                                          hysteresis=hysteresis,
                                          dettype="ed",
                                          n=threshold_trials)
    elif threshold is None:
        raise ValueError("either search_intervals or threshold must be specified")
    cmp = sig.Comparison.from_comparator(stat,
                                         threshold,
                                         hysteresis*threshold)
    eoi = sig.energy_detector_location_estimates(cmp)
    return eoi


def peak_detection(data: sig.Signal, ordc: float):
    """Detect and localise events using a peak detection algorithm."""
    rps = (data.x[1] - data.x[0]) # revs per sample, assuming uniform samples
    fps = rps*ordc # fault occurences per sample
    spf = int(1/fps) # samples per fault occurence
    peaks, _ = scipy.signal.find_peaks(data.y, height=0, distance=spf/2)
    spos = data.x[peaks]
    return spos


def score_med_old(data: sig.Signal,
              initial_filter: npt.ArrayLike,
              ordc: float,
              ordmin: float,
              ordmax: float,
              medfiltsize=100,) -> float:
    """Score MED filtering using on detection metric"""
    # MED stuff
    medfiltest = medest(data.y, initial_filter)
    out = np.convolve(data.y, medfiltest, mode="valid")
    env = abs(scipy.signal.hilbert(out))
    filtsigenv = sig.Signal(env, data.x[medfiltsize-1:],
                            uniform_samples=data.uniform_samples)
    
    # detect and locate
    spos = peak_detection(filtsigenv, ordc)
    _, mag = utl.find_order(spos, ordmin, ordmax)
    score = mag/np.sqrt(len(spos))
    return score, medfiltest


def score_med(signal: sig.Signal,
              filter_length: int, search_intervals: list[tuple[float, float]]) -> Literal["impulse", "step"]:
    """Find the best MED initial conditions given possible faults"""
    # Find best pre-filtering MED filter for initial detection
    initial_shapes = ["impulse", "step"]
    best_score = 0
    for initial_shape in initial_shapes:
        filtered = med_filter(signal, filter_length, initial_shape)
        threshold, score = utl.best_threshold(filtered, search_intervals, hysteresis=0.2, dettype="ed", order_search_density=100)
        if score <= best_score: continue
        results = {
            "score": score,
            "threshold": threshold,
            "filtered": filtered,
            "initial_shape": initial_shape,
        }
    
    return results
