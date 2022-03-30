from __future__ import annotations

import logging
import random
from itertools import product

import numpy as np
import pandas as pd
import brian2 as b2


class MUAPTClassifier:
    """Classify MUAPTs with a Spiking Neural Network (SNN).

    Parameters
    ----------
    n_in: int
        Number of input neurons.
    n_out: int
        Number of output neurons.
    v_rest: float, default=-74
        Rest membrane potential (in mV).
    tau: float, default=20
        Time constant of the neuron (in ms).
    v_th: float, default=-54
        Threshold membrane potential (in mV).
    v_reset: float, default=-60
        membrane potential after reset (in mV).
    e_ex: float, default=0
        Potential of excitatory synapse (in mV).
    tau_ex: float, default=5
        Time constant of the excitatory synapse (in ms).
    g_max: float, default=0.015
        Maximum conductance (adimensional).
    taupre: float, default=20
        Time constant of presynaptic interspike interval (in ms).
    taupost: float, default=20
        Time constant of postsynaptic interspike interval (in ms)
    apre_max: float, default=0.005
        Maximum amount of presynaptic modification (adimensional)
    b_a: float = 1.05
        Ratio between (apost_max * taupost) / (apre_max * taupre).
    ref_period: float, default=5
        Refractory period (in ms).

    Attributes
    ----------
    _n_in: int
        Number of input neurons.
    _n_out: int
        Number of output neurons.
    _g_max: float
        Maximum conductance (adimensional).
    _weight: np.ndarray | None
        Weights of the synapses.
    _eqs_lif: str
        Differential equation describing the evolution of the LIF neuron's membrane potential.
    _eqs_th: str
        Equation describing the behaviour of the LIF neuron when its membrane potential crosses the threshold.
    _eqs_reset: str
        Equation describing the behaviour of the LIF neuron when its membrane potential is reset.
    _eqs_ref: str
        Equation describing the behaviour of the LIF neuron during refractory period.
    _eqs_syn: str
        Differential equation describing the evolution of the synaptic conductivity.
    _eqs_pre: str
        Equation describing the behaviour of the synapse when a presynaptic spike occurs.
    _eqs_post: str
        Equation describing the behaviour of the synapse when a postsynaptic spike occurs.
    """

    def __init__(
            self,
            n_in: int,
            n_out: int,
            v_rest: float = -74,
            tau: float = 20,
            v_th: float = -54,
            v_reset: float = -60,
            e_ex: float = 0,
            tau_ex: float = 5,
            g_max: float = 0.015,
            taupre: float = 20,
            taupost: float = 20,
            apre_max: float = 0.005,
            b_a: float = 1.05,
            ref_period: float = 5
    ) -> None:
        self._n_in = n_in
        self._n_out = n_out
        self._g_max = g_max
        self._weight = None

        # Define equations for LIF neuron
        self._eqs_lif = f"""
            dV/dt = ({v_rest} * mV - V + g_ex * ({e_ex} * mV - V)) / ({tau} * ms) : volt (unless refractory)
            dg_ex/dt = -g_ex / ({tau_ex} * ms) : 1
        """
        self._eqs_th = f"V > {v_th} * mV"
        self._eqs_reset = f"V = {v_reset} * mV"
        self._eqs_ref = f"{ref_period} * ms"

        # Define equations for the synapse
        self._eqs_syn = f"""
            w : 1
            dapre/dt = -apre / ({taupre} * ms) : 1 (event-driven)
            dapost/dt = -apost / ({taupost} * ms) : 1 (event-driven)
        """
        self._eqs_pre = f"""
            g_ex += w
            apre += {apre_max}
            w = clip(w + apost, 0, {g_max})
        """
        self._eqs_post = f"""
            apost += {-apre_max * taupre / taupost * b_a}
            w = clip(w + apre, 0, {g_max})
        """

    @property
    def n_in(self) -> int:
        return self._n_in

    @property
    def n_out(self) -> int:
        return self._n_out

    # noinspection PyTypeChecker
    def train(
            self,
            firings_train: list[pd.DataFrame],
            duration: float,
            epochs: int
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Train the SNN with STDP.

        Parameters
        ----------
        firings_train: list[pd.DataFrame]
            List of DataFrames containing the firings of MUs.
        duration: float
            Duration fo the simulation (in seconds).
        epochs: int
            Number of training epochs.

        Returns
        -------
        hist: dict[str, tuple[np.ndarray, np.ndarray]]
            Dictionary containing the SNN training/inference history.
        """
        # Build plastic SNN with random weights
        n0, n1, syn, spike_mon0, spike_mon1, syn_mon, out_mon = self._build_snn()
        snn = b2.Network(n0, n1, syn, spike_mon0, spike_mon1, syn_mon, out_mon)

        k = 0
        prev_fr0 = 0
        prev_fr1 = 0
        for ep in range(epochs):
            logging.info(f"----- EPOCH {ep + 1} -----")

            # Shuffle list
            firings_train_sh = random.sample(firings_train, len(firings_train))

            for i, firings_train_sample in enumerate(firings_train_sh):
                logging.info(f"Sample {i + 1}/{len(firings_train)}")

                # Set spikes
                n0.set_spikes(
                    firings_train_sample["MU index"].values,
                    (firings_train_sample["Firing time"].values + k) * b2.second
                )
                # Run simulation
                snn.run(duration * b2.second)

                # Check firing rate
                fr0 = np.count_nonzero(spike_mon0.i == 0)
                fr1 = np.count_nonzero(spike_mon1.i == 0)
                logging.info(f"Neuron 0 fired {fr0 - prev_fr0} times.")
                logging.info(f"Neuron 1 fired {fr1 - prev_fr1} times.")
                prev_fr0 = fr0
                prev_fr1 = fr1

                # Increment k
                k += 1

        # Save weight
        self._weight = syn.w

        # Save training history
        hist = {
            "input_spikes": (spike_mon0.t / b2.second, spike_mon0.i),
            "output_spikes": (spike_mon1.t / b2.second, spike_mon1.i),
            "synapses_weights": (syn_mon.t / b2.second, syn_mon.w.T / self._g_max),
            "output_potential": (out_mon.t / b2.second, out_mon.V),
        }
        return hist

    def inference(
            self,
            firings_test_sample: pd.DataFrame,
            duration: float
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Test the SNN.

        Parameters
        ----------
        firings_test_sample: pd.DataFrame
            Sample of a DataFrame containing the firings of MUs.
        duration: float
            Duration fo the simulation (in seconds).

        Returns
        -------
        hist: dict[str, tuple[np.ndarray, np.ndarray]]
            Dictionary containing the SNN training/inference history.
        """
        # Build fixed SNN with learnt weights
        n0, n1, syn, spike_mon0, spike_mon1, syn_mon, out_mon = self._build_snn(plastic=False, w=self._weight)
        snn = b2.Network(n0, n1, syn, spike_mon0, spike_mon1, syn_mon, out_mon)

        # Set spikes
        n0.set_spikes(
            firings_test_sample["MU index"].values,
            firings_test_sample["Firing time"].values * b2.second
        )
        # Run simulation
        snn.run(duration * b2.second)

        # Save training history
        hist = {
            "input_spikes": (spike_mon0.t / b2.second, spike_mon0.i),
            "output_spikes": (spike_mon1.t / b2.second, spike_mon1.i),
            "synapses_weights": (syn_mon.t / b2.second, syn_mon.w.T / self._g_max),
            "output_potential": (out_mon.t / b2.second, out_mon.V),
        }

        return hist

    # noinspection PyTypeChecker
    def _build_snn(
            self,
            plastic: bool = True,
            w: np.ndarray | None = None
    ) -> tuple[
        b2.SpikeGeneratorGroup,
        b2.NeuronGroup,
        b2.Synapses,
        b2.SpikeMonitor,
        b2.SpikeMonitor,
        b2.StateMonitor,
        b2.StateMonitor
    ]:
        """Build an instance of SNN from the given firings.

        Parameters
        ----------
        plastic: bool, default=True
            Whether to make the synapses plastic or fixed.
        w: np.ndarray | None, default=None
            Weight of the synapses.

        Returns
        -------
        n0: b2.SpikeGeneratorGroup
            Input neurons of the network.
        n1: b2.NeuronGroup
            Output neurons of the network.
        syn: b2.Synapses
            Synapses between input and output neurons.
        spike_mon0: b2.SpikeMonitor
            Instance of SpikeMonitor for the input neurons.
        spike_mon1: b2.SpikeMonitor
            Instance of SpikeMonitor for the output neurons.
        syn_mon: b2.StateMonitor
            Instance of StateMonitor for the synaptic weights.
        out_mon: b2.StateMonitor
            Instance of StateMonitor for the output neurons' membrane potential.
        """
        # Define layers
        n0 = b2.SpikeGeneratorGroup(
            self._n_in,
            [0],
            [0] * b2.second
        )
        n1 = b2.NeuronGroup(
            self._n_out,
            self._eqs_lif,
            threshold=self._eqs_th,
            reset=self._eqs_reset,
            refractory=self._eqs_ref,
            method="euler"
        )

        # Define synapses
        conn = list(product(range(self._n_in), range(self._n_out)))
        conn_i, conn_j = zip(*conn)
        syn = b2.Synapses(
            n0,
            n1,
            model=self._eqs_syn if plastic else "w : 1",
            on_pre=self._eqs_pre if plastic else "g_ex += w",
            on_post=self._eqs_post if plastic else None,
            method="linear"
        )
        syn.connect(i=conn_i, j=conn_j)
        syn.w = f"rand() * {self._g_max}" if w is None else w

        # Define monitors
        spike_mon0 = b2.SpikeMonitor(n0)
        spike_mon1 = b2.SpikeMonitor(n1)
        syn_mon = b2.StateMonitor(syn, "w", record=True)
        out_mon = b2.StateMonitor(n1, "V", record=True)

        return n0, n1, syn, spike_mon0, spike_mon1, syn_mon, out_mon
