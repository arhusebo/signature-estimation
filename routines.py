import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.stats
import faultevent.event as evt
import faultevent.signal as sig
from faultevent import util as utl


def irfs(data: sig.Signal,
         spos1: npt.ArrayLike,
         ordmin: float,
         ordmax: float,
         sigsize: int = 400,
         sigshift: int = -150,
         enedet_max_loc_error: int = 10,
         n_iter: int = 10,
         threshold_trials = 10,) -> np.ndarray:
    

    # initial iteration
    ordf1, _ = evt.find_order(spos1, ordmin, ordmax)
    mu1, kappa1 = evt.fit_vonmises(ordf1, spos1)
    z1 = evt.map_circle(ordf1, spos1)
    crt1 = scipy.stats.vonmises.pdf(z1, kappa1, loc=mu1)
    idx1 = np.argsort(crt1)
    signat1 = utl.estimate_signature(data, spos1[idx1], crt1[idx1],
                                    sigsize, max_error = enedet_max_loc_error,
                                    n0 = sigshift)

    # ith iteration
    det1 = sig.MatchedFilterEnvelopeDetector(signat1)
    stat1 = det1.statistic(data)
    hys1 = .2
    norm1 = np.linalg.norm(signat1)
    thr1 = utl.best_threshold(stat1, ordmin, ordmax, hys=hys1,
                              n=threshold_trials)/norm1
    det_list = [det1]
    for i in range(1, n_iter-1):
        stat_i = det_list[i-1].statistic(data)
        thr_i = thr1*np.linalg.norm(det_list[i-1].h)
        cmp_i = sig.Comparison.from_comparator(stat_i, thr_i, thr_i*hys1)
        spos_i = np.asarray(sig.matched_filter_location_estimates(cmp_i))
        ordf_i, _ = utl.find_order(spos_i, ordmin, ordmax)
        mu_i, kappa_i = evt.fit_vonmises(ordf_i, spos_i)
        z_i = evt.map_circle(ordf_i, spos_i)
        crt_i = scipy.stats.vonmises.pdf(z_i, kappa_i, loc=mu_i)
        idx_sorted = np.argsort(crt_i)
        sig_i = utl.estimate_signature(data, spos_i[idx_sorted],
                                        crt_i[idx_sorted],
                                        len(det_list[i-1].h))
        det_i = sig.MatchedFilterEnvelopeDetector(sig_i)
        det_list.append(det_i)
    return sig_i, ordf_i, mu_i, kappa_i


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
            ordmin: float,
            ordmax: float,
            enedetsize: int = 50,
            threshold_trials: int = 10) -> np.ndarray:
    """Detect and return locations of events using an energy detector"""
    det = sig.EnergyDetector(enedetsize)
    stat = det.statistic(data)
    hys = .8
    thr = utl.best_threshold(stat, ordmin, ordmax, hys=hys, dettype="ed",
                              n=threshold_trials)
    cmp = sig.Comparison.from_comparator(stat, thr, hys*thr)
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


def score_med(data: sig.Signal,
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