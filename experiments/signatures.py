import numpy as np

import faultevent.signal as sig
import faultevent.event as evt

import algorithms

import gsim
from gsim.gfigure import GFigure


def signature_experiment(sigsize, sigshift, resid, ordc, medfiltsize) -> GFigure:
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

    G = GFigure(yaxis=irfs_result.sigest, xaxis=resid.x[:sigsize])

    ords_plot = np.linspace(0, 10, 1000)
    evspec_plot = evt.event_spectrum(ords_plot, irfs_result.eosp)
    
    G_spectrum = GFigure()
    G_spectrum.next_subplot(
        xaxis=ords_plot,
        yaxis=abs(evspec_plot),
        ylabel="Event spectrum magnitude"
    )
    G_spectrum.add_curve(
        xaxis=[irfs_result.ordf, irfs_result.ordf],
        yaxis=[0, len(irfs_result.eosp)],
        legend="Fault order",
        styles=["--"],
    )
    G_spectrum.add_curve(
        xaxis=[ords_plot[0], ords_plot[-1]],
        yaxis=[len(irfs_result.eosp), len(irfs_result.eosp)],
        legend="#Detected events",
        styles=["--"]
    )

    return [G, G_spectrum]


class ExperimentSet(gsim.AbstractExperimentSet):

    def experiment_1001(l_args):
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

        G = signature_experiment(sigsize = 400,
                                 sigshift = -150,
                                 resid = resid,
                                 ordc = 6.7087166,
                                 medfiltsize = 100,)
        return G
    
    def experiment_1002(l_args):
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

        G = signature_experiment(sigsize = 400,
                                 sigshift = -150,
                                 resid = resid,
                                 ordc = 6.7087166,
                                 medfiltsize = 100,)
        return G
    
    def experiment_1003(l_args):
        from data.unsw import UNSWDataLoader
        from data import unsw_path
        dl = UNSWDataLoader(unsw_path)
        mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
        #mf = dl["Test 1/6Hz/vib_000356575_06.mat"]
        mf = dl["Test 1/6Hz/vib_000002663_06.mat"]
        angfhz = 6
        fs = 51200
        signalt = mf.vib
        model = sig.ARModel.from_signal(mh.vib[:10000], 41)
        residt = model.residuals(signalt)
        resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

        G = signature_experiment(sigsize = 200,
                                 sigshift = -100,
                                 resid = resid,
                                 ordc = 3.56,
                                 medfiltsize = 100,)
        return G
    

    def experiment_1004(l_args):
        from data.unsw import UNSWDataLoader
        from data import unsw_path
        dl = UNSWDataLoader(unsw_path)
        mh = dl["Test 1/6Hz/vib_000002663_06.mat"]
        mf = dl["Test 4/Multiple speeds/vib_001674651_06.mat"]
        angfhz = 6
        fs = 51200
        signalt = mf.vib
        model = sig.ARModel.from_signal(mh.vib[:10000], 41)
        residt = model.residuals(signalt)
        resid = sig.Signal.from_uniform_samples(residt.y, angfhz/fs)

        G = signature_experiment(sigsize = 200,
                                 sigshift = -100,
                                 resid = resid,
                                 ordc = 5.42,
                                 medfiltsize = 100,)
        return G
    

    def experiment_1005(l_args):
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

        G = signature_experiment(sigsize = 400,
                                 sigshift = -150,
                                 resid = resid,
                                 ordc = 5.4152,
                                 medfiltsize = 100,)
        return G
    

    def experiment_1006(l_args):
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

        G = signature_experiment(sigsize = 400,
                                 sigshift = -150,
                                 resid = resid,
                                 ordc = 4.7135,
                                 medfiltsize = 100,)
        return G
    

    def experiment_1007(l_args):
        n_cols = 3
        ttl = ["a", "b", "c", "d", "e", "f"]
        import matplotlib
        import matplotlib.pyplot as plt
        for i in range(6):
            print(ExperimentSet.load_GFigures(1001+i)[0].l_subplots[0])
        G = GFigure(figsize=(3.5, 2.0),
                    num_subplot_columns=n_cols)
        G.l_subplots = [ExperimentSet.load_GFigures(1001+i)[0].l_subplots[0] for i in range(6)]

        for i, subplt in enumerate(G.l_subplots):
            subplt.ylabel = ""
            #subplt.title = ttl[i]
            if i>=len(G.l_subplots)-n_cols: subplt.xlabel = "Revs"
            for curve in subplt.l_curves:
                curve.style = "k-"

        with matplotlib.rc_context({"lines.linewidth" : .5,
                                    "font.size": 8}):
            fig = G.plot()
        ax = fig.get_axes()
        for i in range(len(ax)):
            ax[i].grid(visible=False, which="both", axis="both")
            ax[i].set_yticks([])
            #ax[i].title.set_size(7)
            ax[i].annotate(ttl[i], (.8, .7), xycoords="axes fraction")
        plt.tight_layout()

        plt.show()



    def experiment_1008(l_args):
            import matplotlib.pyplot as plt
            from data.cwru import CWRUDataLoader
            from data import cwru_path
            dl = CWRUDataLoader(cwru_path)
            mh = dl[100]
            mf = dl[175]
            rpm = dl.info["175"]["rpm"]
            fs = 51200
            
            plt.figure()
            plt.psd(mh.vib.y, label="healthy", Fs=fs)
            plt.psd(mf.vib.y, label="faulty", Fs=fs)
            plt.legend()
            plt.show()
    

    def experiment_1009(l_args):
        import matplotlib.pyplot as plt
        y = ExperimentSet.load_GFigures(1005)[0].l_subplots[0].l_curves[0].yaxis

        spec = np.fft.rfft(y)
        freq = np.fft.rfftfreq(len(y), d=1/51200)
        plt.plot(freq, 20*np.log10(abs(spec)))
        plt.show()
    
    def experiment_1010(l_args):
        import matplotlib.pyplot as plt
        from scipy.signal import hilbert
        from data.cwru import CWRUDataLoader
        from data import cwru_path
        dl = CWRUDataLoader(cwru_path)
        mh = dl[100]
        mf = dl[175]
        rpm = dl.info["175"]["rpm"]
        fs = 51200

        model = sig.ARModel.from_signal(mh.vib[:10000], 75)
        
        residth = model.residuals(mh.vib)
        residtf = model.residuals(mf.vib)

        residtfenv = np.abs(hilbert(residtf.y))

        plt.figure()
        plt.psd(residth.y, label="healthy", Fs=fs)
        plt.psd(residtf.y, label="faulty", Fs=fs)
        plt.psd(residtfenv, label="faulty envelope", Fs=fs)
        plt.legend()

        _, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(residtf.y)
        ax[1].plot(residtfenv)
        plt.show()
    

    def experiment_1011(l_args):
        import matplotlib.pyplot as plt
        from scipy.signal import hilbert
        y = ExperimentSet.load_GFigures(1005)[0].l_subplots[0].l_curves[0].yaxis

        env = hilbert(y)
        plt.plot(env.real)
        plt.plot(abs(env))
        plt.show()