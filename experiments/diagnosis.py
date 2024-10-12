import logging
from multiprocessing import Pool

from algorithms import score_med, enedetloc, diagnose_fault
from faultevent.signal import Signal, ARModel

from simsim import experiment, presentation

output_path = "results/diagnosis"


def diagnosis(signal: Signal, faults, medfiltsize):
    prefilt = score_med(signal, medfiltsize, faults)
    spos = enedetloc(prefilt["filtered"], search_intervals=faults.values())
    fault_name, ordf1 = diagnose_fault(spos, faults)
    return fault_name



def sample_experiment(data):
    info = data["info"]
    dl = data["dataloader"]
    model = data["model"]
    faults = data["faults"]

    mf = dl[info["id"]]
    signalt = mf.vib # signal in time domain
    residt = model.residuals(signalt) # AR residuals in time domain

    # Angular speed of these measurements are approximately constant,
    # no resampling is applied.
    #signal = Signal.from_uniform_samples(signalt.y, (info["rpm"]/60)/dl.info["fs"])
    resid = Signal.from_uniform_samples(residt.y, (info["rpm"]/60)/dl.info["fs"])
    prefilt = score_med(resid, 100, faults)
    spos = enedetloc(prefilt["filtered"], search_intervals=faults.values())
    fault_name = diagnose_fault(spos, faults, 0.0)

    return fault_name


@experiment(output_path)
def ex_cwru():

    from data import cwru
    from data import cwru_path
    dl = cwru.CWRUDataLoader(cwru_path)
    mh = dl["100"]
    
    model = ARModel.from_signal(mh.vib[:10000], 75) # AR model


    faults = {
        cwru.Diagnostics.ROLLER: [4.7135+d for d in (-0.1, 0.1)],
        cwru.Diagnostics.INNER: [5.4152+d for d in (-0.1, 0.1)],
        cwru.Diagnostics.OUTER: [3.5848+d for d in (-0.1, 0.1)],
    }

    
    total_samples = len(dl.info["data"])

    process_data = []
    for i, info in enumerate(dl.info["data"]):
        data = {
            "dataloader": dl,
            "info": info,
            "model": model,
            "faults": faults,
        }
        process_data.append(data)

    with Pool() as p:
        diagnosis_results = p.map(sample_experiment, process_data)
    
    results = [{
        "id": info["id"],
        "diagnosis": cwru.Diagnostics.HEALTHY if not diag else diag,
    } for info, diag in zip(dl.info["data"], diagnosis_results)]

    return results


@presentation(ex_cwru)
def present_cwru_diagnosis(results):
    from data import cwru
    from data import cwru_path
    dl = cwru.CWRUDataLoader(cwru_path)
    print(r"\begin{center}")
    print(r"\begin{tabular}{ c c c }")
    print(r"\hline")
    print(" & ".join(["Actual", "Diagnosed", "Size (mm)"]), r"\\")
    print(r"\hline")
    for result in results:
        signal_id, diagres = result.values()
        signal_info = dl.signal_info(signal_id)
        actual, fsize = cwru.fault(signal_info["name"])
        print(" & ".join([actual, diagres, str(fsize)]), r"\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{center}")