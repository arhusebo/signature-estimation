from string import ascii_lowercase
from collections import deque

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import faultevent.signal as sig
import faultevent.event as evt

import algorithms
import data

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

    score_med_results = algorithms.score_med(resid, medfiltsize, [(ordmin, ordmax)])
    residf = score_med_results["filtered"]

    # IRFS method.
    spos1 = algorithms.enedetloc(residf, search_intervals=[(ordmin, ordmax)])
    #spos1 = algorithms.enedetloc(residf,
    #                             threshold=score_med_results["threshold"])
    irfs = algorithms.irfs(resid, spos1, ordmin, ordmax, sigsize, sigshift)
    irfs_result, = deque(irfs, maxlen=1)

    return irfs_result


@experiment(output_path)
def uia1():
    dl = data.dataloader("uia")
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
    dl = data.dataloader("uia")
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
    dl = data.dataloader("unsw")
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
    dl = data.dataloader("unsw")
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
    dl = data.dataloader("cwru")
    mh = dl["100"]
    mf = dl["175"]
    rpm = dl.signal_info("175")["rpm"]
    fs = dl.info["fs"]
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
    dl = data.dataloader("cwru")
    mh = dl["100"]
    mf = dl["192"]
    rpm = dl.signal_info("192")["rpm"]
    fs = dl.info["fs"]
    signalt = mf.vib
    model = sig.ARModel.from_signal(mh.vib[:10000], 75)
    residt = model.residuals(signalt)
    resid = sig.Signal.from_uniform_samples(residt.y, (rpm/60)/fs)

    return signature_experiment(sigsize = 400,
                                sigshift = -150,
                                resid = resid,
                                ordc = 4.7135,
                                medfiltsize = 100,), fs, rpm


@presentation(uia1, uia2, unsw1, unsw2, cwru1, cwru2)
def present_signatures(list_results: list[tuple[algorithms.IRFSIteration, float, float]]):
    n_cols = 3
    matplotlib.rcParams.update({"font.size": 6})
    fig, ax = plt.subplots(2, 3, figsize=(3.5, 2.0))
    for i, results in enumerate(list_results):
        irfs_result, fs, rpm = results
        row = i//n_cols
        col = i%n_cols
        axc = ax[row][col] # current axes
        revs = np.arange(len(irfs_result["sigest"]))*rpm/60/fs
        axc.plot(revs, irfs_result["sigest"], c="k", lw=0.5)
        axc.set_yticks([])
        axc.annotate(ascii_lowercase[i], (.8, .8), xycoords="axes fraction")
        if row==1: axc.set_xlabel("Revs")

    plt.tight_layout() 
    plt.show()
