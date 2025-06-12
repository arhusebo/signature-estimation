from typing import TypedDict, Callable, NotRequired
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import scipy.signal

from faultevent.signal import Signal


class FaultDescriptor(TypedDict):
    name: NotRequired[str]
    ord: float
    std: float
    signature: Callable[[int], npt.ArrayLike]


class AnomalyDescriptor(TypedDict):
    amount: int
    signature: Callable[[int], npt.ArrayLike]


class InterferenceDescriptor(TypedDict):
    sir: float
    central_frequency: float
    bandwidth: float


class SignalDescriptor(TypedDict):
    length: int
    sample_frequency: float
    shaft_frequency: float
    faults: Sequence[FaultDescriptor]
    snr: NotRequired[float]
    snir: NotRequired[float]
    anomaly: NotRequired[AnomalyDescriptor]
    interference: NotRequired[InterferenceDescriptor]


def avg_fault_period(desc: SignalDescriptor,
                         fault_index: int = 0) -> float:
    """Computes the average number of shaft revolutions between fault event"""
    return (desc["sample_frequency"]
            /desc["shaft_frequency"])/desc["faults"][fault_index]["ord"]


#common_fault_signature = lambda n: (n>=0)*np.sinc(n/8+1)
DEFAULT_ANOMALY_SIGNATURE = lambda n: (n>=0)*np.sinc(n/2+1)
def DEFAULT_FAULT_SIGNATURE(n):
    return (n>=0)*np.sinc(n/8+1)

def sigtilde(signat: Callable[[int], float], sigloc: Sequence[int], n: int):
    """Samples a train of signatures given signature function 'signat'
    evaluated at 0, 1, ..., 'n' with signature offsets 'sigloc'."""
    return np.sum([signat(np.arange(n)-n0) for n0 in sigloc], axis=0)


def generate_signal(desc: SignalDescriptor, seed=None):
    """Generates a residual signal according to the provided
    ResidualDescriptor. Always generates the same result unless a
    different value of 'seed' is used.
    
    Returns a dictionary containing the signal, event
    labels and EOSPs.
    """
    # TODO: Support signal filter, e.g. AR
    rng = np.random.default_rng(seed)
    signal = np.zeros((desc["length"]), dtype=float)
    eosp_ = []
    elbl_ = []

    # Generate fault signatures
    for i, fault in enumerate(desc["faults"]):
        df = (desc["sample_frequency"]
              /desc["shaft_frequency"])/fault["ord"]
        eosp = np.arange(0, desc["length"], df, dtype=float)
        eosp += rng.standard_normal(len(eosp))*fault["std"]
        signal += sigtilde(fault["signature"], eosp, desc["length"])

        eosp_.append(eosp)
        elbl_.append(np.ones_like(eosp, dtype=int)+i)

    # Generate anomaly signatures
    if anomaly := desc.get("anomaly", None):
        eosp = rng.uniform(0, desc["length"], anomaly["amount"])
        signal += sigtilde(anomaly["signature"], eosp, desc["length"])

        eosp_.append(eosp)
        elbl_.append(np.zeros_like(eosp, dtype=int))
    
    
    sigpow = np.var(signal) # total signatures power

    # Generate signal noise
    if snr := desc.get("snr", None):
        noisepow = sigpow/snr
        noise = rng.standard_normal(desc["length"]) * np.sqrt(noisepow)
    else:
        noise = np.zeros_like(signal)

    # Generate random interference componen
    if interference := desc.get("interference", None):
        Wn_low = interference["central_frequency"] - interference["bandwidth"]/2
        Wn_high = interference["central_frequency"] + interference["bandwidth"]/2
        interf = rng.laplace(0, 1, signal.shape)
        interf = np.sign(interf)*interf**2
        interf_sos = scipy.signal.butter(4, Wn=(Wn_low, Wn_high),
                                         btype="bandpass",
                                         output="sos",
                                         fs=desc["sample_frequency"],)
        interf = scipy.signal.sosfilt(interf_sos, interf)
        intpow = sigpow/interference["sir"]
        interf *= np.sqrt(intpow)/np.std(interf)
    else:
        interf = np.zeros_like(signal)

    ni = noise + interf # noise and interference
    if (snir := desc.get("snir", None)) and (snr or interference):
        # SNIR = var(resid) / var(noise + interf)
        # var(noise + interf) = var(resid) / SNIR
        # std(noise + interf) = sqrt(var(resid) / SNIR)
        # std(noise + interf) = std(resid) / sqrt(SNIR)
        pow_ni = sigpow/snir
        ni *= np.sqrt(pow_ni)/np.std(ni)
    
    signal += ni

    # Instantiate and return signal object
    dx = 1/(desc["sample_frequency"]/desc["shaft_frequency"])
    out = Signal.from_uniform_samples(signal, dx)
    event_shaft_positions = np.concatenate(eosp_)*dx
    event_labels = np.concatenate(elbl_)
    return {
        "signal": out,
        "eosp": event_shaft_positions,
        "event_labels": event_labels,
    }
