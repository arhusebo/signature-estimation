import numpy as np
import scipy.signal

import faultevent.signal as sig
import faultevent.event as evt
import faultevent.util as utl

import algorithms

import gsim
from gsim.gfigure import GFigure


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

def update_scores_gfigure(G, methodname, ndets, mags):
    """Updates a GFigure with the score of method 'methodname'."""
    mag = mags
    frac = mag/ndets
    #idx_best = np.argmax(mag/np.sqrt(ndets))
    G.add_curve(xaxis=ndets,
                yaxis=frac,
                legend=methodname,)


def benchmark_experiment(sigsize, sigshift, signal, resid, ordc,
                         medfiltsize, sknperseg,
                         use_irfs_eosp = False) -> GFigure:
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

    # Residuals are pre-filtered using MED.
    initial_filters = np.zeros((2,medfiltsize), dtype=float)
    # Impulse condition:
    initial_filters[0, medfiltsize//2] = 1
    initial_filters[0, medfiltsize//2+1] = -1
    # Step condition:
    initial_filters[1, :medfiltsize//2] = 1
    initial_filters[1, medfiltsize//2:] = -1

    # Find best pre-filtering MED filter for initial detection
    scores = np.zeros((len(initial_filters),), dtype=float)
    medfilts = np.zeros_like(initial_filters)
    for i, initial_filter in enumerate(initial_filters):
        scores[i], medfilts[i] = algorithms.score_med(resid,
                                                    initial_filter,
                                                    ordc,
                                                    ordmin,
                                                    ordmax,)
    residf = algorithms.medfilt(resid, medfilts[np.argmax(scores)])

    # IRFS method.
    spos1 = algorithms.enedetloc(residf, ordmin, ordmax) # First iteration
    irfs_result = algorithms.irfs(resid, spos1, ordmin, ordmax, sigsize, sigshift)

    if use_irfs_eosp:
        irfs_val_to_sort = irfs_result.magnitude * irfs_result.certainty
        irfs_idx = np.argsort(irfs_val_to_sort)[::-1]
        irfs_nvals = len(irfs_result.eosp)
        irfs_ndets = np.arange(irfs_nvals)+1
        irfs_mags = np.zeros((irfs_nvals,), dtype=float)
        for i in range(irfs_nvals):
            spos = irfs_result.eosp[irfs_idx][:i+1]
            irfs_mags[i] = abs(evt.event_spectrum(irfs_result.ordf, spos))

    else:
        irfs_out = np.correlate(resid.y, irfs_result.sigest, mode="valid")
        irfs_filt = sig.Signal(irfs_out, resid.x[:-len(irfs_result.sigest)+1],
                            resid.uniform_samples)
        def irfs_weight(spos):
            z = evt.map_circle(irfs_result.ordf, spos)
            u = scipy.stats.vonmises.pdf(z, irfs_result.kappa, loc=irfs_result.mu)
            return u
        irfs_ndets, irfs_mags = detect_and_sort(irfs_filt, ordc, ordmin, ordmax, weightfunc=irfs_weight)
    
    print("IRFS done.")

    # MED method. Signal is filtered using filter obtained by MED.
    medfiltest = algorithms.medest(signal.y, initial_filters[0])
    med_filt = algorithms.medfilt(signal, medfiltest)
    med_ndets, med_mags = detect_and_sort(med_filt, ordc, ordmin, ordmax)
    print("MED done.")

    # AR-MED method. Residuals are filtered using filter obtained by AR-MED.
    armedfiltest = algorithms.medest(resid.y, initial_filters[0])
    armed_filt = algorithms.medfilt(resid, armedfiltest)
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

    G = GFigure(xlabel="Detected events",
                ylabel="Fraction of true positives")
    update_scores_gfigure(G, "IRFS", irfs_ndets, irfs_mags)
    update_scores_gfigure(G, "MED", med_ndets, med_mags)
    update_scores_gfigure(G, "AR-MED", armed_ndets, armed_mags)
    update_scores_gfigure(G, "SK", sk_ndets, sk_mags)
    update_scores_gfigure(G, "AR-SK", arsk_ndets, arsk_mags)
    n_events_max = ordc*signal.x[-1]
    G.add_curve(xaxis=[n_events_max]*2, yaxis=[0, 1],
                legend="Max events", styles=["k--"])

    G_sigest = GFigure(xaxis=resid.x[:len(irfs_result.sigest)],
                       yaxis=irfs_result.sigest,
                       xlabel="Revs",
                       ylabel="Signature estimate")
    
    G_filt = GFigure()
    G_filt.next_subplot(xaxis=irfs_filt.x, yaxis=irfs_filt.y, ylabel="IRFS")
    G_filt.next_subplot(xaxis=med_filt.x, yaxis=med_filt.y, ylabel="MED")
    G_filt.next_subplot(xaxis=armed_filt.x, yaxis=armed_filt.y, ylabel="AR-MED")
    G_filt.next_subplot(xaxis=sk_filt.x, yaxis=sk_filt.y, ylabel="SK")
    G_filt.next_subplot(xaxis=arsk_filt.x, yaxis=arsk_filt.y, ylabel="AR-SK")

    return [G, G_sigest, G_filt]


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

        G = benchmark_experiment(sigsize = 400,
                                 sigshift = -150,
                                 signal = signal,
                                 resid = resid,
                                 ordc = 6.7087166, # contact angle corrected
                                 medfiltsize = 100,
                                 sknperseg = 1000,)
        return G
    
    def experiment_1002(l_args):
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

        G = benchmark_experiment(sigsize = 200,
                                 sigshift = -100,
                                 signal = signal,
                                 resid = resid,
                                 #ordc = 3.56,
                                 ordc = 5.42,
                                 medfiltsize = 100,
                                 sknperseg = 256,)
        return G

    def experiment_1003(l_args):

        from data.cwru import CWRUDataLoader
        from data import cwru_path
        dl = CWRUDataLoader(cwru_path)
        mh = dl[100]
        mf = dl[175]
        
        rpm = dl.info["175"]["rpm"] # angular frequency in Hz
        fs = 48e3 # sample frequency
        signalt = mf.vib # signal in time domain
        model = sig.ARModel.from_signal(mh.vib[:10000], 75) # AR model
        residt = model.residuals(signalt) # AR residuals in time domain

        # Angular speed of these measurements are approximately constant,
        # no resampling is applied.
        signal = sig.Signal.from_uniform_samples(signalt.y, (rpm/60)/fs)
        resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

        G = benchmark_experiment(sigsize = 400,
                                 sigshift = -150,
                                 signal = signal,
                                 resid = resid,
                                 ordc = 5.4152,
                                 medfiltsize = 100,
                                 sknperseg = 256,)
        return G
    
    def experiment_1004(l_args):
        """This experiment combines the results of all experiments into
        a single GFigure"""
        import matplotlib
        import matplotlib.pyplot as plt

        l_G_uia = ExperimentSet.load_GFigures(1001)
        l_G_unsw = ExperimentSet.load_GFigures(1002)
        l_G_cwru = ExperimentSet.load_GFigures(1003)
        G = GFigure(figsize=(3.5, 4.0))
        G.l_subplots = l_G_uia[0].l_subplots +\
                       l_G_unsw[0].l_subplots +\
                       l_G_cwru[0].l_subplots
        
        G_sigest = GFigure(num_subplot_columns=3, figsize=(3.5, 1.0))
        G_sigest.l_subplots = l_G_uia[1].l_subplots +\
                              l_G_unsw[1].l_subplots +\
                              l_G_cwru[1].l_subplots
        
        # edit loaded GFigure
        datalabels = ["UiA", "UNSW", "CWRU"]
        for i, subplt in enumerate(G.l_subplots):
            #subplt.l_curves[0], subplt.l_curves[-1] = subplt.l_curves[-1], subplt.l_curves[0]
            subplt.ylabel = f"True positive rate\n({datalabels[i]})"
            subplt.xlabel = ""
        G.l_subplots[-1].xlabel = "Detections"

        for i, subplt in enumerate(G_sigest.l_subplots):
            subplt.ylabel = ""
            subplt.xlabel = "Revs"
            for curve in subplt.l_curves:
                curve.style = "k-"
            
        G_sigest.l_subplots[0].ylabel = "Signature\nestimate"

        # draw GFigure and customise plotting
        matplotlib.rcParams.update({"font.size": 8})
        fig = G.plot()
        ax = fig.get_axes()
        ax[0].legend(ncol=3, bbox_to_anchor=(0.5, 1.6), loc="upper center")
        ax[1].get_legend().remove()
        ax[2].get_legend().remove()
        plt.tight_layout()
        
        with matplotlib.rc_context({"lines.linewidth" : .5}):
            fig_sigest = G_sigest.plot()
        ax = fig_sigest.get_axes()
        for i in range(len(ax)):
            ax[i].grid(visible=False, which="both", axis="both")
            ax[i].set_yticks([])
        plt.tight_layout()

        plt.show()