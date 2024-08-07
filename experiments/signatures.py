from string import ascii_lowercase

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import faultevent.signal as sig
import faultevent.event as evt

import algorithms

from simsim import experiment, presentation


output_path = "results/signature"


def signature_experiment(sigsize, sigshift, resid, ordc, medfiltsize):
    """Wrapper function of a general experiment to test all benchmark methods
    on one set of data.
    
    Arguments:
    sigsize -- size of the signature
    sigshift -- 
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

    # Residuals are pre-filtered using .
    initial_filters = np.zeros((2,medfiltsize), dtype=float)
    # impulse
    initial_filters[0, medfiltsize//2] = 1
    initial_filters[0, medfiltsize//2+1] = -1
    # step
    initial_filters[1, :medfiltsize//2] = 1
    initial_filters[1, medfiltsize//2:] = -1

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
    spos1 = algorithms.enedetloc(residf, ordmin, ordmax)
    irfs_result = algorithms.irfs(resid, spos1, ordmin, ordmax, sigsize, sigshift)

    return irfs_result


@experiment(output_path)
def uia1():
    from data.uia import UiADataLoader
    from data import uia_path
    dl = UiADataLoader(uia_path)
    mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
    mf = dl["y2016-m09-d24/00-40-22 1000rpm - 51200Hz - 100LOR.h5"]
    rpm = 1000
    fs = 51200
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 117)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return signature_experiment(sigsize = 400,
                                sigshift = -150,
                                resid = resid,
                                ordc = 6.7087166,
                                medfiltsize = 100,), fs, rpm


@experiment(output_path)
def uia2():
    from data.uia import UiADataLoader
    from data import uia_path
    dl = UiADataLoader(uia_path)
    mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
    mf = dl["y2016-m09-d24/00-50-31 1000rpm - 51200Hz - 100LOR.h5"]
    rpm = 1000
    fs = 51200
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 117)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return signature_experiment(sigsize = 400,
                                sigshift = -150,
                                resid = resid,
                                ordc = 6.7087166,
                                medfiltsize = 100,), fs, rpm


@experiment(output_path)
def unsw1():
    from data.unsw import UNSWDataLoader
    from data import unsw_path
    dl = UNSWDataLoader(unsw_path)
    mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
    #mf = dl["Test 1/6Hz/vib_000356575_06.mat"]
    mf = dl["Test 1/6Hz/vib_000002663_06.mat"]
    angfhz = 6
    rpm = angfhz*60 
    fs = 51200
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 41)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

    return signature_experiment(sigsize = 200,
                                sigshift = -100,
                                resid = resid,
                                ordc = 3.56,
                                medfiltsize = 100,), fs, rpm


@experiment(output_path)
def unsw2():
    from data.unsw import UNSWDataLoader
    from data import unsw_path
    dl = UNSWDataLoader(unsw_path)
    mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
    mf = dl["Test 4/Multiple speeds/vib_001674651_06.mat"]
    angfhz = 6
    rpm = angfhz*60
    fs = 51200
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 41)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

    return signature_experiment(sigsize = 200,
                                sigshift = -100,
                                resid = resid,
                                ordc = 5.42,
                                medfiltsize = 100,), fs, rpm


@experiment(output_path)
def cwru1():
    from data.cwru import CWRUDataLoader
    from data import cwru_path
    dl = CWRUDataLoader(cwru_path)
    mh = dl[100]
    mf = dl[175]
    rpm = dl.info["175"]["rpm"]
    fs = 51200
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 75)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return signature_experiment(sigsize = 400,
                                sigshift = -150,
                                resid = resid,
                                ordc = 5.4152,
                                medfiltsize = 100,), fs, rpm

@experiment(output_path)
def cwru2():
    from data.cwru import CWRUDataLoader
    from data import cwru_path
    dl = CWRUDataLoader(cwru_path)
    mh = dl[100]
    mf = dl[192]
    rpm = dl.info["192"]["rpm"]
    fs = 48e3
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 75)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return signature_experiment(sigsize = 400,
                                sigshift = -150,
                                resid = resid,
                                ordc = 4.7135,
                                medfiltsize = 100,)


@presentation(output_path, ["uia1", "uia2", "unsw1", "unsw2", "cwru1", "cwru2"])
def present_signatures(list_results: list[tuple[algorithms.IRFSResult, float, float]]):
    n_cols = 3
    fig, ax = plt.subplots(2, 3)
    for i, result in enumerate(list_results):
        irfs_result, fs, rpm = result
        row = i//n_cols
        col = i%n_cols
        axc = ax[row][col] # current axes
        revs = np.arange(len(irfs_result.sigest))*rpm/60/fs
        axc.plot(revs, irfs_result.sigest, c="k")
        axc.set_yticks([])
        axc.annotate(ascii_lowercase[i], (.8, .7), xycoords="axes fraction")
        if row==1: axc.set_xlabel("Revs")

    plt.tight_layout() 
    plt.show()
