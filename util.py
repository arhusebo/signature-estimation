"""Utility class where we throw everything that does not fit anywhere else"""

from functools import cache

from faultevent.signal import ARModel
import ml
import ml2
import data


@cache
def get_armodel(dataset: data.DataName) -> ARModel:
    """Fit an AR model for generating signals from synthetic residuals"""
    dl = data.dataloader(dataset)
    match dataset:
        case data.DataName.UIA:
            mh = dl["y2016-m09-d20/00-13-28 1000rpm - 51200Hz - 100LOR.h5"]
            return ARModel.from_signal(mh.vib[:10000], 117) # ar model
        case data.DataName.UNSW:
            mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
            return ARModel.from_signal(mh.vib[:10000], 41)
        case data.DataName.CWRU:
            mh = dl["100"]
            return ARModel.from_signal(mh.vib[:10000], 75)

@cache
def get_mlmodel(dataset: data.DataName) -> ml.MLSignalModel:
    return ml.MLSignalModel(ml.Model.load(ml.model_filepath(dataset)))


@cache
def get_ml2model(dataset: data.DataName) -> ml2.ML2SignalModel:
    """The direct fault-component denoiser (ml2)."""
    return ml2.load_model(dataset)
