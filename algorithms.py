from collections import namedtuple
from typing import Literal
import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.stats
import faultevent.event as evt
import faultevent.signal as sig
from faultevent import util as utl

IRFSResult = namedtuple("IRFSResult",
                        ["sigest", "eosp", "magnitude", "certainty", "ordf", "mu", "kappa"])

def irfs(data: sig.Signal,
         spos1: npt.ArrayLike,
         ordmin: float,
         ordmax: float,
         sigsize: int,
         sigshift: int,
         hys: float = .01,
         enedet_max_loc_error: int = 10,
         n_iter: int = 10,
         threshold_trials = 10,) -> np.ndarray:
    

    # initial iteration
    ordf1, _ = evt.find_order(spos1, ordmin, ordmax)
    mu1, kappa1 = evt.fit_vonmises(ordf1, spos1)
    z1 = evt.map_circle(ordf1, spos1)
    crt1 = scipy.stats.vonmises.pdf(z1, kappa1, loc=mu1)
    idx1 = np.argsort(crt1)[::-1]
    signat1 = utl.estimate_signature(data,
                                     sigsize,
                                     x=spos1[idx1],
                                     weights=crt1[idx1],
                                     max_error = enedet_max_loc_error,
                                     n0 = sigshift)

    # ith iteration
    det1 = sig.MatchedFilterEnvelopeDetector(signat1)
    stat1 = det1.statistic(data)
    norm1 = np.linalg.norm(signat1)
    thr1, _ = utl.best_threshold(stat1, [(ordmin, ordmax)], hysteresis=hys,
                              n=threshold_trials)/norm1
    det_list = [det1]
    for i in range(1, n_iter-1):
        stat_i = det_list[i-1].statistic(data)
        thr_i = thr1*np.linalg.norm(det_list[i-1].h)
        cmp_i = sig.Comparison.from_comparator(stat_i, thr_i, thr_i*hys)
        spos_i, mag_i = np.asarray(sig.matched_filter_location_estimates(cmp_i))
        ordf_i, _ = utl.find_order(spos_i, ordmin, ordmax)
        mu_i, kappa_i = evt.fit_vonmises(ordf_i, spos_i)
        z_i = evt.map_circle(ordf_i, spos_i)
        crt_i = scipy.stats.vonmises.pdf(z_i, kappa_i, loc=mu_i)
        idx_sorted = np.argsort(crt_i)[::-1]
        sig_i = utl.estimate_signature(data,
                                       len(det_list[i-1].h),
                                       x=spos_i[idx_sorted],
                                       weights=crt_i[idx_sorted])
        det_i = sig.MatchedFilterEnvelopeDetector(sig_i)
        det_list.append(det_i)
    result = IRFSResult(sig_i, spos_i, mag_i, crt_i, ordf_i, mu_i, kappa_i)
    return result


def diagnose_fault(eosp: npt.ArrayLike,
                   faults: dict,
                   fault_threshold: float = 0.0,
                   ) -> tuple[float, float] | None:
    """Diagnose the fault type given event shaft positions and a list
    of dictionaries containing fault characteristics. Fault
    dictionaries must be on the form
    {
        "OR" : (4.9, 5.1),
        "IR" : (3.7, 3.8)
    }
    """

    best_score = 0.0
    results = None
    for key, interval in faults.items():
        _, magnitude = evt.find_order(eosp, *interval, density=1e3)
        score = magnitude/np.sqrt(len(eosp)) if len(eosp) else 0.0
        if score > best_score >= fault_threshold:
            results = key
            best_score = score
    return results


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
              search_intervals: list[tuple[float, float]],
              enedetsize: int = 50,
              hysteresis: float = .8,
              threshold: float | None = None,
              threshold_trials: int = 10) -> np.ndarray:
    """Detect and return locations of events using an energy detector"""
    det = sig.EnergyDetector(enedetsize)
    stat = det.statistic(data)
    if threshold is None:
        threshold, _ = utl.best_threshold(stat,
                                          search_intervals,
                                          hysteresis=hysteresis,
                                          dettype="ed",
                                          n=threshold_trials)
    cmp = sig.Comparison.from_comparator(stat,
                                         threshold,
                                         hysteresis*threshold)
    spos = np.asarray(sig.energy_detector_location_estimates(cmp))
    return spos


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
              filter_length: int, faults: dict) -> Literal["impulse", "step"]:
    """Find the best MED initial conditions given possible faults"""
    # Find best pre-filtering MED filter for initial detection
    initial_shapes = ["impulse", "step"]
    search_intervals = faults.values()
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