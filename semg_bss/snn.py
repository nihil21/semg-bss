from __future__ import annotations

import pandas as pd
from brian2 import *


class MUAPTClassifier:
    """Classify MUAPTs with a Spiking Neural Network (SNN).

    Parameters
    ----------
    n_in: int
        Number of input neurons.
    p: float, default=0.2
        Degree of sparseness in the synapse (between 0 and 1)

    Attributes
    ----------
    _snn: Network
        Instance of the SNN.
    _n0: SpikeGeneratorGroup
        Input of the SNN.
    _syn: Synapses
        The synapses of the SNN.
    _monitor: SpikeMonitor
        Instance of SpikeMonitor.
    """

    # noinspection PyTypeChecker
    def __init__(self, n_in: int, p: float = 0.2) -> None:
        assert 0 < p <= 1, "The degree of sparseness should be between 0 and 1."

        # Create network
        self._snn = Network()

        # Define constants
        V_rest = -74 * mV
        E_ex = 0 * mV
        tau = 20 * ms
        tau_ex = 5 * ms
        V_th = -54 * mV
        V_reset = -60 * mV
        taupre = 20 * ms
        taupost = 20 * ms
        g_max = 0.02
        Apre = 0.001
        Apost = -Apre * taupre / taupost * 1.05

        # Define differential equations
        eqs = """
            dV/dt = (V_rest - V + g_ex * (E_ex - V)) / tau : volt
            dg_ex/dt = -g_ex / tau_ex : 1
        """
        eqs_syn = """
            w : 1
            dapre/dt = -apre / taupre : 1 (event-driven)
            dapost/dt = -apost / taupost : 1 (event-driven)
        """
        eqs_pre = """
            g_ex += w
            apre += Apre
            w = clip(w + apost, 0, g_max)
        """
        eqs_post = """
            apost += Apost
            w = clip(w + apre, 0, g_max)
        """

        # Define layers
        self._n0 = SpikeGeneratorGroup(n_in, [0], [0] * ms)
        n1 = NeuronGroup(2, eqs, threshold="V > V_th", reset="V = V_reset")

        # Define synapse
        self._syn = Synapses(self._n0, n1, model=eqs_syn, on_pre=eqs_pre, on_post=eqs_post, method="linear")
        self._syn.connect(condition="i != j", p=p)
        self._syn.w = "rand() * g_max"

        # Define spike monitor
        self._monitor = SpikeMonitor(self._n0)

        # Add everything to the network
        self._snn.add(self._n0)
        self._snn.add(n1)
        self._snn.add(self._syn)
        self._snn.add(self._monitor)

    @property
    def syn(self) -> Synapses:
        return self._syn

    @property
    def monitor(self) -> SpikeMonitor:
        return self._monitor

    def set_spikes(self, firings: pd.DataFrame) -> None:
        """Set the spikes of the input neurons.

        Parameters
        ----------
        firings: pd.DataFrame
            Firing times for each MU.
        """
        self._n0.set_spikes(
            firings["MU index"].values,
            firings["Firing time"].values * second
        )

    def run(self, duration: float) -> None:
        """Run the simulation.

        Parameters
        ----------
        duration: float
            Duration fo the simulation (in seconds).
        """
        self._snn.run(duration * second)
