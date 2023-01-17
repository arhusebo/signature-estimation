import numpy as np

from faultevent.signal import ARModel

import gsim
from gsim.gfigure import GFigure

def general_fitting_experiment(sigfit, sigval, prange):
    """For every AR order in 'prange', fit an AR model on training signal
    and calculate the AIC by predicting on validation signal"""
    aic = np.zeros_like(prange, dtype=float)
    for i, p in enumerate(prange):
        model = ARModel.from_signal(sigfit, p)
        pred = model.residuals(sigval)
        var = np.var(pred.y)
        aic[i] = 2*(p+1)+(len(sigval)-p)*np.log(var)
        #print(f"{i+1}/{len(prange)}")
    
    G = GFigure(xaxis=prange,
                yaxis=aic,
                xlabel="p",
                ylabel="AIC")
    return G

class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
        print("Fitting AR model to data from UIA")
        from data.uia import UiADataLoader
        nsampfit = 10000 # number of samples to use for fitting model
        dl = UiADataLoader("../andreas_data/ALT Bearing 2 h5/")
        sigfit = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"].vib[:nsampfit]
        sigval = dl["y2016-m09-d20/00-23-37 1000rpm - 51200Hz - 100LOR.h5"].vib[:nsampfit]

        prange = np.arange(1, 300)
        G = general_fitting_experiment(sigfit, sigval, prange)
        return G
    
    def experiment_1002(l_args):
        print("Fitting AR model to data from UNSW")
        from data.unsw import UNSWDataLoader
        nsampfit = 10000 # number of samples to use for fitting model
        dl = UNSWDataLoader("../unsw_dataset/Test 1/")
        sigfit = dl["6Hz/vib_000002663_06.mat"].vib[:nsampfit]
        sigval = dl["6Hz/vib_000005667_06.mat"].vib[:nsampfit]

        prange = np.arange(1, 300)
        prange = np.arange(1, 300)
        G = general_fitting_experiment(sigfit, sigval, prange)
        return G
    
    def experiment_1003(l_args):
        print("Fitting AR model to data from CWRU")
        from data.cwru import CWRUDataLoader
        nsampfit = 10000 # number of samples to use for fitting model
        dl = CWRUDataLoader("../case_dataset/info.json")
        sigfit = dl[100].vib[:nsampfit]
        sigval = dl[100].vib[-nsampfit:]

        prange = np.arange(1, 300)
        prange = np.arange(1, 300)
        G = general_fitting_experiment(sigfit, sigval, prange)
        return G