import numpy as np
import scipy.signal

import faultevent.signal as sig
import faultevent.event as evt
import faultevent.util as utl

import routines

import gsim
from gsim.gfigure import GFigure


def detect_and_sort(filt: sig.Signal, ordc, ordmin, ordmax, maxevents=10000):
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
    idx_sorted = np.argsort(properties["peak_heights"])[::-1]
    spos = filt.x[peaks]
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


def update_scores_gfigure(G, methodname, ndets, mags):
    """Updates a GFigure with the score of method 'methodname'."""
    mag = mags
    frac = mag/ndets
    #idx_best = np.argmax(mag/np.sqrt(ndets))
    G.add_curve(xaxis=ndets,
                yaxis=frac,
                legend=methodname,)


def general_experiment(**kwargs) -> GFigure:
    """Wrapper function of a general experiment to test all benchmark methods
    on one set of data.
    
    Keyword arguments:
    signal -- faultevent.signal.Signal object containing vibrations in shaft domain
    resid -- faultevent.signal.Signal object containing AR residuals in shaft domain
    fs -- sample frequency
    ordc -- characteristic fault order
    medfiltsize -- MED filter size
    sknbands -- number of frequency bands for SK estimation
    
    """
    signal = kwargs["signal"]
    resid = kwargs["resid"]
    fs = kwargs["fs"]
    ordc = kwargs["ordc"]
    medfiltsize = kwargs["medfiltsize"]
    sknbands = kwargs["sknbands"]

    ordmin = ordc-.5
    ordmax = ordc+.5

    # IRFS method. Residuals are filtered using matched filter.
    irfs_sigest = routines.irfs(resid, ordmin, ordmax)
    irfs_out = np.correlate(resid.y, irfs_sigest, mode="valid")
    irfs_filt = sig.Signal(irfs_out, resid.x[:-len(irfs_sigest)+1],
                        resid.uniform_samples)
    irfs_ndets, irfs_mags = detect_and_sort(irfs_filt, ordc, ordmin, ordmax)
    print("IRFS done.")

    # MED method. Signal is filtered using filter obtained by MED.
    med_filt = routines.medfilt(signal, medfiltsize)
    med_ndets, med_mags = detect_and_sort(med_filt, ordc, ordmin, ordmax)
    print("MED done.")

    # AR-MED method. Residuals are filtered using filter obtained by AR-MED.
    armed_filt = routines.medfilt(resid, medfiltsize)
    armed_ndets, armed_mags = detect_and_sort(armed_filt, ordc, ordmin, ordmax)
    print("AR-MED done.")

    # SK method. Signal is filtered using filter maximising SK.
    sk_filt = routines.skfilt(signal, sknbands, fs)
    sk_ndets, sk_mags = detect_and_sort(sk_filt, ordc, ordmin, ordmax)
    print("SK done.")

    # AR-SK method. Residuals are filtered using filter maximising SK.
    arsk_filt = routines.skfilt(resid, sknbands, fs)
    arsk_ndets, arsk_mags = detect_and_sort(arsk_filt, ordc, ordmin, ordmax)
    print("AR_SK done.")

    G = GFigure(xlabel="Detected events",
                ylabel="Fraction of true positives")
    update_scores_gfigure(G, "IRFS", irfs_ndets, irfs_mags)
    update_scores_gfigure(G, "MED", med_ndets, med_mags)
    update_scores_gfigure(G, "AR-MED", armed_ndets, armed_mags)
    update_scores_gfigure(G, "SK", sk_ndets, sk_mags)
    update_scores_gfigure(G, "AR-SK", arsk_ndets, arsk_mags)

    G_sigest = GFigure(xaxis=resid.x[:len(irfs_sigest)],
                       yaxis=irfs_sigest,
                       xlabel="Revs",
                       ylabel="Signature estimate")
    return [G, G_sigest]


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):

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

        G = general_experiment(signal = signal,
                               resid = resid,
                               fs = fs,
                               ordc = 6.7087166, # contact angle corrected
                               medfiltsize = 100,
                               sknbands = 100,)
        return G
    
    def experiment_1002(l_args):
        from data.unsw import UNSWDataLoader
        from data import unsw_path
        dl = UNSWDataLoader(unsw_path)
        mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
        mf = dl["Test 1/Multiple speeds/vib_000330272_20.mat"]
        
        angfhz = 20 # angular frequency in Hz
        fs = 51200 # sample frequency
        signalt = mf.vib # signal in time domain
        model = sig.ARModel.from_signal(mh.vib[:10000], 41) # AR model
        residt = model.residuals(signalt) # AR residuals in time domain

        # Angular speed of these measurements are approximately constant,
        # no resampling is applied.
        signal = sig.Signal.from_uniform_samples(signalt.y, angfhz/fs)
        resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

        G = general_experiment(signal = signal,
                               resid = resid,
                               fs = fs,
                               ordc = 3.56,
                               medfiltsize = 100,
                               sknbands = 100,)
        return G

    def experiment_1003(l_args):

        from data.cwru import CWRUDataLoader
        from data import cwru_path
        dl = CWRUDataLoader(cwru_path)
        mh = dl[100]
        mf = dl[112]
        
        rpm = dl.info["112"]["rpm"] # angular frequency in Hz
        fs = 51200 # sample frequency
        signalt = mf.vib # signal in time domain
        model = sig.ARModel.from_signal(mh.vib[:10000], 75) # AR model
        residt = model.residuals(signalt) # AR residuals in time domain

        # Angular speed of these measurements are approximately constant,
        # no resampling is applied.
        signal = sig.Signal.from_uniform_samples(signalt.y, (rpm/60)/fs)
        resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

        G = general_experiment(signal = signal,
                               resid = resid,
                               fs = fs,
                               ordc = 5.4152,
                               medfiltsize = 100,
                               sknbands = 100,)
        return G
    
    def experiment_1004(l_args):
        """This experiment combines the results of all experiments into
        a single GFigure"""
        l_G_uia = ExperimentSet.load_GFigures(1001)
        l_G_unsw = ExperimentSet.load_GFigures(1002)
        l_G_cwru = ExperimentSet.load_GFigures(1003)
        G = GFigure(figsize=(5.5, 10.0))
        G.l_subplots = l_G_uia[0].l_subplots +\
                       l_G_unsw[0].l_subplots +\
                       l_G_cwru[0].l_subplots
        
        G_sigest = GFigure(num_subplot_columns=3)
        G_sigest.l_subplots = l_G_uia[1].l_subplots +\
                       l_G_unsw[1].l_subplots +\
                       l_G_cwru[1].l_subplots
        return [G, G_sigest]