import numpy as np
import numpy.typing as npt
import scipy.signal
import scipy.stats
import faultevent.event as evt
import faultevent.signal as sig
from faultevent import util as utl


def irfs(data: sig.Signal, ordmin: float, ordmax: float,
         enedetsize: int=50, sigsize: int = 400, sigshift: int = -150,
         enedet_max_loc_error: int = 10, n_iter: int = 10,
         threshold_trials = 10) -> np.ndarray:
    
    # 1st iteration
    det1 = sig.EnergyDetector(enedetsize)
    stat1 = det1.statistic(data)
    hys1 = .8
    thr1 = utl.best_threshold(stat1, ordmin, ordmax, hys=hys1, dettype="ed",
                              n=threshold_trials)
    cmp1 = sig.Comparison.from_comparator(stat1, thr1, hys1*thr1)
    spos1 = np.asarray(sig.energy_detector_location_estimates(cmp1))
    ordf1, _ = evt.find_order(spos1, ordmin, ordmax)
    mu1, kappa1 = evt.fit_vonmises(ordf1, spos1)
    z1 = evt.map_circle(ordf1, spos1)
    crt1 = scipy.stats.vonmises.pdf(z1, kappa1, loc=mu1)
    idx1 = np.argsort(crt1)
    sig1 = utl.estimate_signature(data, spos1[idx1], crt1[idx1],
                                    sigsize, max_error = enedet_max_loc_error,
                                    n0 = sigshift)
    
    # ith iteration
    det2 = sig.MatchedFilterEnvelopeDetector(sig1)
    stat2 = det2.statistic(data)
    hys2 = .2
    norm2 = np.linalg.norm(sig1)
    thr2 = utl.best_threshold(stat2, ordmin, ordmax, hys=hys2)/norm2
    det_list = [det2]
    for i in range(1, n_iter-1):
        stat_i = det_list[i-1].statistic(data)
        thr_i = thr2*np.linalg.norm(det_list[i-1].h)
        cmp_i = sig.Comparison.from_comparator(stat_i, thr_i, thr_i*hys2)
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
    return sig_i

def medest(x: npt.ArrayLike, f0: npt.ArrayLike, its: int=10) -> np.ndarray:
    """Minimum entropy deconvolution (R. A. Wiggins, 1978).
    Given input x, estimates the filter (FIR) that maximises the output
    kurtosis (and effectively impulsiveness), where f0 is the initial guess."""
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

def medfilt(signal: sig.Signal, n: int):
    """Estimates MED filter og size n and filters the signal"""
    # initialisation
    medfilt0 = np.zeros((n,), dtype=float)
    medfilt0[n//2] = 1.0
    medfilt0[n//2] = -1.0
    medfiltest = medest(signal.y, medfilt0)
    out = np.convolve(signal.y, medfiltest, mode="valid") # filtering
    return sig.Signal(out, signal.x[n-1:],
                        uniform_samples=signal.uniform_samples)

def cwt_wrapper(x: npt.ArrayLike, n: int, fs=1.0, w=5.0):
    """Wrapper for working with the Morlet wavelet and frequencies"""
    freqs = np.linspace(1.0, fs/2, n)
    widths = w*fs / (2*freqs*np.pi)
    cwt = scipy.signal.cwt(x, scipy.signal.morlet2, widths, w=w,
                           dtype=np.complex128)
    return cwt, freqs

def sk_cwt(x: npt.ArrayLike, n: int = 10, fs=1.0, w=5.0):
    """Returns spectral kurtosis estimates using the CWT"""
    cwt, freqs = cwt_wrapper(x, n, fs, w)
    sk = scipy.stats.kurtosis(abs(cwt), axis=-1)
    return sk, freqs

def skfilt(signal: sig.Signal, n: int = 10, fs=1.0, w=5.0):
    """Filters the signal using the CWT band of highest kurtosis"""
    cwt, _ = cwt_wrapper(signal.y, n, fs, w)
    sk = scipy.stats.kurtosis(abs(cwt), axis=-1)
    idxmax = np.argmax(sk)
    out = cwt[idxmax]
    return sig.Signal(out, signal.x,
                      uniform_samples=signal.uniform_samples)