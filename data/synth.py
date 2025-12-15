from typing import TypedDict, Callable, NotRequired
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from faultevent.signal import Signal

import data

class FaultDescriptor(TypedDict):
    name: NotRequired[str]
    ord: float
    std: float
    snr: NotRequired[float]
    signature: Callable[[int], npt.ArrayLike]


class AnomalyDescriptor(TypedDict):
    amount: int
    signature: Callable[[int], npt.ArrayLike]
    snr: NotRequired[float]


class HealthyComponentDescriptor(TypedDict):
    dataname: data.DataName
    signal_id: str


class VibrationDescriptor(TypedDict):
    length: int
    sample_frequency: float
    shaft_frequency: float
    healthy_component: HealthyComponentDescriptor
    faults: Sequence[FaultDescriptor]
    anomaly: NotRequired[AnomalyDescriptor]


def avg_fault_period(desc: VibrationDescriptor, fault_index: int = 0) -> float:
    """Computes the average number of shaft revolutions between fault event"""
    return (desc["sample_frequency"]
            /desc["shaft_frequency"])/desc["faults"][fault_index]["ord"]


@np.vectorize
def signt_stpres(f, tau, t):
    return (np.exp(-t/(3*tau))*(-np.cos(2*np.pi*(f/6)*t))
            +np.exp(-t/(5*tau))) if t >= 0.0 else 0.0

@np.vectorize
def signt_impres(f, tau, t):
    return np.exp(-t/tau)*np.sin(2*np.pi*f*t) if t>=0.0 else 0.0


def signt_res(f, tau, d, t, fs=1.0):
    """t in terms of samples"""
    return signt_stpres(f, tau, t/fs)/20 + signt_impres(f, tau, (t-d)/fs)


def DEFAULT_ANOMALY_SIGNATURE(n):
    return (n>=0)*np.sinc(n/2+1)


def DEFAULT_FAULT_SIGNATURE(n, d=30):
    return signt_res(6.5e3, 0.001, d, n, fs=25.e3)


def sigtilde(sigloc: Sequence[int], n: int, sig_samp = None, sig_func: Callable[[int], float] = None, fs=1.0):
    """Samples a train of signatures given signature function 'signat'
    evaluated at 0, 1, ..., 'n' with signature offsets 'sigloc'."""
    assert sig_samp or sig_func
    if sig_samp:
        out = np.zeros((n,), dtype=float)
        for loc in sigloc:
            idx0 = max(0, loc)
            idx1 = min(n, loc+len(sig_samp))
            out[idx0:idx1] = sig_samp[:idx0-idx1]
        return 
    elif sig_func:
        return np.sum([sig_func(np.arange(n)-n0) for n0 in sigloc], axis=0)


def signature_train(eosp: Sequence[float], signature, signal_length, fs, fshaft):
    """Create a train of signatures"""
    out = np.zeros((signal_length,), dtype=float)
    n = np.array((eosp/fshaft)*fs, dtype=int)
    for idx in n:
        idx0 = max(0, idx)
        idx1 = min(signal_length, idx+len(signature))
        slicelen = idx1-idx0
        out[idx0:idx1] = signature[:slicelen]
    return out


@dataclass
class VibrationData:
    desc: VibrationDescriptor
    signal: npt.NDArray[np.float64]
    eosp: npt.NDArray[np.float64]
    event_labels: npt.NDArray[np.int_]



def generate_vibration(desc: VibrationDescriptor, seed=0) -> VibrationData:
    """Generates a residual signal according to the provided
    ResidualDescriptor. Always generates the same result unless a
    different value of 'seed' is used.
    """
    rng = np.random.default_rng(seed)

    # Noise (healthy) component
    # A random slice is extracted from the signal `signal_id` of dataset
    # `dataname` and used as the noise component of the synthetic signal.
    dl = data.dataloader(desc["healthy_component"]["dataname"])
    noise_full = dl[desc["healthy_component"]["signal_id"]].vib.y
    idx0 = rng.choice(len(noise_full)-desc["length"])
    noise = noise_full[idx0:idx0+desc["length"]]
    pow_noise = np.var(noise)


    signal = np.zeros((desc["length"]), dtype=float)
    eosp_ = []
    elbl_ = []

    eosp_end = desc["length"]/desc["sample_frequency"]*desc["shaft_frequency"]
    # Generate fault signatures
    for i, fault in enumerate(desc["faults"]):
        eosp = np.arange(0, eosp_end, 1/fault["ord"])
        eosp += rng.standard_normal(len(eosp))*fault["std"]
        component = signature_train(eosp, fault["signature"], desc["length"],
                                 fs=desc["sample_frequency"],
                                 fshaft=desc["shaft_frequency"])

        eosp_.append(eosp)
        elbl_.append(np.ones_like(eosp, dtype=int)+i)

        if snr := fault.get("snr", None):
            assert snr > 0.0
            pow_component = pow_noise*snr
            component = np.sqrt(pow_component)*component/np.std(fault["signature"])


        signal += component
            

    # Generate anomaly signatures
    if anomaly := desc.get("anomaly", None):
        eosp = rng.uniform(0, eosp_end, anomaly["amount"])
        component = signature_train(eosp, anomaly["signature"], desc["length"],
                                 fs=desc["sample_frequency"],
                                 fshaft=desc["shaft_frequency"])

        eosp_.append(eosp)
        elbl_.append(np.zeros_like(eosp, dtype=int))
        
        if snr := anomaly.get("snr", None):
            assert snr > 0.0
            pow_component = pow_noise*snr
            component = np.sqrt(pow_component)*component/np.std(anomaly["signature"])
    
        signal += component

    # Instantiate and return signal object
    dx = 1/(desc["sample_frequency"]/desc["shaft_frequency"])
    out = Signal.from_uniform_samples(noise + signal, dx)
    event_shaft_positions = np.concatenate(eosp_)#*dx
    event_labels = np.concatenate(elbl_)
    return VibrationData(desc=desc, signal=out, eosp=event_shaft_positions,
                         event_labels=event_labels)
