"""Copyright 2022 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import pandas as pd
import brian2 as b2


def visualise_connectivity(syn):
    ns = len(syn.source)
    nt = len(syn.target)
    b2.figure(figsize=(10, 4))
    b2.subplot(121)
    b2.plot(b2.zeros(ns), b2.arange(ns), "ok", ms=10)
    b2.plot(b2.ones(nt), b2.arange(nt), "ok", ms=10)
    for i, j in zip(syn.i, syn.j):
        b2.plot([0, 1], [i, j], "-k")
    b2.xticks([0, 1], ["Source", "Target"])
    b2.ylabel("Neuron index")
    b2.xlim(-0.1, 1.1)
    b2.ylim(-1, max(ns, nt))
    b2.subplot(122)
    b2.plot(syn.i, syn.j, "ok")
    b2.xlim(-1, ns)
    b2.ylim(-1, nt)
    b2.xlabel("Source neuron index")
    b2.ylabel("Target neuron index")


class MUAPTClassifierSNN:
    """Classify MUAPTs with a Spiking Neural Network (SNN).

    Parameters
    ----------
    n_inp : int
        Number of input neurons.
    n_pool : int
        Number of neurons in the pool.
    classes : list of str
        List of classes to recognize.
    v_rest_e : float, default=-70.
        Rest membrane potential for excitatory neurons (in mV).
    v_rest_i : float, default=-70.
        Rest membrane potential for inhibitory neurons (in mV).
    v_reset_e : float, default=-60.
        Membrane potential after reset for excitatory neurons (in mV).
    v_reset_i : float, default=-60.
        Membrane potential after reset for inhibitory neurons (in mV).
    v_thres_e : float, default=-54.
        Threshold membrane potential of excitatory neurons (in mV).
    v_thres_i : float, default=-54.
        Threshold membrane potential of inhibitory neurons (in mV).
    refrac_e : float, default=5.
        Refractory period of excitatory neurons (in ms).
    refrac_i : float, default=5.
        Refractory period of inhibitory neurons (in ms).
    w_max : float, default=0.015
        Maximum synaptic conductance (adimensional).
    v_exc : float, default=0.
        Equilibrium potential of excitatory synapses (in mV).
    v_inh : float, default=-70.
        Equilibrium potential of inhibitory synapses (in mV).
    tau_exc : float, default=20.
        Time constant of membrane potential for excitatory neurons (in ms).
    tau_inh : float, default=20.
        Time constant of membrane potential for inhibitory neurons (in ms).
    tau_ge : float, default=5.
        Time constant of conductance for excitatory synapses (in ms).
    tau_gi : float, default=5.
        Time constant of conductance for inhibitory synapses (in ms).
    tau_pre : float, default=20.
        Time constant of pre-synaptic trace (in ms).
    tau_post : float, default=20.
        Time constant of post-synaptic trace (in ms).
    max_pot : float, default=0.005
        Maximum potentiation (adimensional).
    b_a : float, default=1.05
        Ratio between maximum depression and maximum potentiation.

    Attributes
    ----------
    _n_inp : int
        Number of input neurons.
    _n_pool : int
        Number of neurons in the pool.
    _classes : list of str
        List of classes to recognize.
    _v_rest_e : float
        Rest membrane potential for excitatory neurons (in mV).
    _v_rest_i : float
        Rest membrane potential for inhibitory neurons (in mV).
    _v_reset_e : float
        Membrane potential after reset for excitatory neurons (in mV).
    _v_reset_i : float
        Membrane potential after reset for inhibitory neurons (in mV).
    _v_thres_e : float
        Threshold membrane potential of excitatory neurons (in mV).
    _v_thres_i : float
        Threshold membrane potential of inhibitory neurons (in mV).
    _refrac_e : float
        Refractory period of excitatory neurons (in ms).
    _refrac_i : float
        Refractory period of inhibitory neurons (in ms).
    _w_max : float
        Maximum synaptic conductance (adimensional).
    _v_exc : float
        Equilibrium potential of excitatory synapses (in mV).
    _v_inh : float
        Equilibrium potential of inhibitory synapses (in mV).
    _tau_exc : float
        Time constant of membrane potential for excitatory neurons (in ms).
    _tau_inh : float
        Time constant of membrane potential for inhibitory neurons (in ms).
    _tau_ge : float
        Time constant of conductance for excitatory synapses (in ms).
    _tau_gi : float
        Time constant of conductance for inhibitory synapses (in ms).
    _tau_pre : float
        Time constant of pre-synaptic trace (in ms).
    _tau_post : float
        Time constant of post-synaptic trace (in ms).
    _max_pot : float
        Maximum potentiation (adimensional).
    _max_dep : float
        Maximum depression (adimensional).
    _weights_inp2exc : ndarray or None
        Weights between input and excitatory neurons.
    _weights_exc2inh : ndarray or None
        Weights between excitatory and inhibitory neurons.
    _weights_inh2exc : ndarray or None
        Weights between inhibitory and excitatory neurons.
    _delays_inp2exc : ndarray or None
        Delays of the excitatory synapses.
    _readout : dict of {int, str}
        Dictionary representing the association between output neuron and gesture.
    """

    def __init__(
            self,
            n_inp: int,
            n_pool: int,
            classes: list[str],
            v_rest_e: float = -70.,
            v_rest_i: float = -70.,
            v_reset_e: float = -60.,
            v_reset_i: float = -60.,
            v_thres_e: float = -54.,
            v_thres_i: float = -54.,
            refrac_e: float = 5.,
            refrac_i: float = 5.,
            w_max: float = 0.015,
            v_exc: float = 0.,
            v_inh: float = -70.,
            tau_exc: float = 20.,
            tau_inh: float = 20.,
            tau_ge: float = 5.,
            tau_gi: float = 5.,
            tau_pre: float = 20.,
            tau_post: float = 20.,
            max_pot: float = 0.005,
            b_a: float = 1.05
    ) -> None:
        # Neuron parameters
        self._n_inp = n_inp
        self._n_pool = n_pool
        self._v_rest_e = v_rest_e
        self._v_rest_i = v_rest_i
        self._v_reset_e = v_reset_e
        self._v_reset_i = v_reset_i
        self._v_thres_e = v_thres_e
        self._v_thres_i = v_thres_i
        self._refrac_e = refrac_e
        self._refrac_i = refrac_i
        self._w_max = w_max
        self._v_exc = v_exc
        self._v_inh = v_inh
        self._tau_exc = tau_exc
        self._tau_inh = tau_inh
        self._tau_ge = tau_ge
        self._tau_gi = tau_gi
        self._tau_pre = tau_pre
        self._tau_post = tau_post
        self._max_pot = max_pot
        self._max_dep = max_pot * tau_pre / tau_post * b_a
        # Classes
        self._classes = classes
        # Weights and delays
        self._weights_inp2exc = None
        self._weights_exc2inh = None
        self._weights_inh2exc = None
        self._delays_inp2exc = None
        # Readout
        self._readout: dict[int, str] = {}

    @property
    def n_inp(self) -> int:
        return self._n_inp

    @property
    def n_pool(self) -> int:
        return self._n_pool

    def train(
            self,
            firings_train: list[tuple[pd.DataFrame, str, float]],
            epochs: int = 1
    ) -> dict[str, dict[Any, Any]]:
        """Train the SNN with STDP.

        Parameters
        ----------
        firings_train : list of tuple of (DataFrame, str, float)
            List of tuples containing the firings of MUs, the label of the gesture and the duration (in seconds).
        epochs : int, default=1
            Number of epochs.

        Returns
        -------
        dict of {str, dict}
            Dictionary containing the SNN training/inference history.
        """
        # Build fixed SNN
        neuron_groups, synapses, monitors = self._build_snn(plastic=True)
        snn = b2.Network(neuron_groups, synapses, monitors)

        # Training history
        hist = {
            "spikes_inp": {},
            "spikes_exc": {},
            "spikes_inh": {},
            "V_exc": {},
            "V_inh": {},
            # "w_inp2exc": {}
        }
        # Firing statistics
        firing_stats: dict[int, dict[str, int]] = {}
        for n in range(self._n_pool):
            firing_stats[n] = {c: 0 for c in self._classes}

        wait = 1.  # seconds

        prev_spikes = np.zeros(shape=(self._n_pool,), dtype=int)
        cur_sim_time = 0
        tot_iter = 0
        for ep in range(epochs):
            logging.info(f"----- Epoch {ep + 1} -----")

            # Shuffle list
            firings_train_sh = random.sample(firings_train, len(firings_train))
            for i, (firings_train_x, firings_train_y, duration) in enumerate(firings_train_sh):
                logging.info(
                    f"Sample {i + 1}/{len(firings_train)} - Gesture {firings_train_y} - Duration: {duration:.2f} s"
                )

                # Set input spikes
                neuron_groups["inp"].set_spikes(
                    firings_train_x["MU index"].values,
                    (firings_train_x["Firing time"].values + cur_sim_time) * b2.second,
                )

                # Run simulation
                snn.run((duration + wait) * b2.second)

                # Update firing statistics
                spikes = np.array([np.count_nonzero(monitors["spikes_exc"].i == n) for n in range(self._n_pool)])
                cur_spikes = spikes - prev_spikes
                for n in range(self._n_pool):
                    firing_stats[n][firings_train_y] += cur_spikes[n]
                prev_spikes = spikes

                # Read data from monitors and produce training history:
                # 1. Input spikes
                mon_state = monitors["spikes_inp"].get_states(["t", "i"], units=False)
                idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                if firings_train_y not in hist["spikes_inp"]:
                    hist["spikes_inp"][firings_train_y] = {}
                    hist["spikes_inp"][firings_train_y]["t"] = mon_state["t"][idx]
                    hist["spikes_inp"][firings_train_y]["i"] = mon_state["i"][idx]
                else:
                    hist["spikes_inp"][firings_train_y]["t"] = np.concatenate(
                        [hist["spikes_inp"][firings_train_y]["t"], mon_state["t"][idx]]
                    )
                    hist["spikes_inp"][firings_train_y]["i"] = np.concatenate(
                        [hist["spikes_inp"][firings_train_y]["i"], mon_state["i"][idx]]
                    )
                # 2. Spikes of excitatory neurons
                mon_state = monitors["spikes_exc"].get_states(["t", "i"], units=False)
                idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                if firings_train_y not in hist["spikes_exc"]:
                    hist["spikes_exc"][firings_train_y] = {}
                    hist["spikes_exc"][firings_train_y]["t"] = mon_state["t"][idx]
                    hist["spikes_exc"][firings_train_y]["i"] = mon_state["i"][idx]
                else:
                    hist["spikes_exc"][firings_train_y]["t"] = np.concatenate(
                        [hist["spikes_exc"][firings_train_y]["t"], mon_state["t"][idx]]
                    )
                    hist["spikes_exc"][firings_train_y]["i"] = np.concatenate(
                        [hist["spikes_exc"][firings_train_y]["i"], mon_state["i"][idx]]
                    )
                # 3. Spikes of inhibitory neurons
                mon_state = monitors["spikes_inh"].get_states(["t", "i"], units=False)
                idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                if firings_train_y not in hist["spikes_inh"]:
                    hist["spikes_inh"][firings_train_y] = {}
                    hist["spikes_inh"][firings_train_y]["t"] = mon_state["t"][idx]
                    hist["spikes_inh"][firings_train_y]["i"] = mon_state["i"][idx]
                else:
                    hist["spikes_inh"][firings_train_y]["t"] = np.concatenate(
                        [hist["spikes_inh"][firings_train_y]["t"], mon_state["t"][idx]]
                    )
                    hist["spikes_inh"][firings_train_y]["i"] = np.concatenate(
                        [hist["spikes_inh"][firings_train_y]["i"], mon_state["i"][idx]]
                    )
                # 4. Potential of excitatory neurons
                mon_state = monitors["V_exc"].get_states(["t", "V"], units=False)
                idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                if not hist["V_exc"]:
                    hist["V_exc"]["t"] = mon_state["t"][idx]
                    hist["V_exc"]["V"] = mon_state["V"][idx]
                    hist["V_exc"]["thres"] = self._v_thres_e / 1000 * np.ones_like(mon_state["t"][idx])
                else:
                    hist["V_exc"]["t"] = np.concatenate(
                        [hist["V_exc"]["t"], mon_state["t"][idx]]
                    )
                    hist["V_exc"]["V"] = np.concatenate(
                        [hist["V_exc"]["V"], mon_state["V"][idx]]
                    )
                    hist["V_exc"]["thres"] = np.concatenate(
                        [hist["V_exc"]["thres"], self._v_thres_e / 1000 * np.ones_like(mon_state["t"][idx])]
                    )
                # 5. Potential of inhibitory neurons
                mon_state = monitors["V_inh"].get_states(["t", "V"], units=False)
                idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                if not hist["V_inh"]:
                    hist["V_inh"]["t"] = mon_state["t"][idx]
                    hist["V_inh"]["V"] = mon_state["V"][idx]
                    hist["V_inh"]["thres"] = self._v_thres_i / 1000 * np.ones_like(mon_state["t"][idx])
                else:
                    hist["V_inh"]["t"] = np.concatenate(
                        [hist["V_inh"]["t"], mon_state["t"][idx]]
                    )
                    hist["V_inh"]["V"] = np.concatenate(
                        [hist["V_inh"]["V"], mon_state["V"][idx]]
                    )
                    hist["V_inh"]["thres"] = np.concatenate(
                        [hist["V_inh"]["thres"], self._v_thres_i / 1000 * np.ones_like(mon_state["t"][idx])]
                    )
                # 6. Synaptic weights (from input to excitatory)
                # mon_state = monitors["w_inp2exc"].get_states(["t", "w"], units=False)
                # idx = np.flatnonzero(mon_state["t"] >= cur_sim_time)
                # if not hist["w_inp2exc"]:
                #     hist["w_inp2exc"]["t"] = mon_state["t"][idx]
                #     hist["w_inp2exc"]["w"] = mon_state["w"][idx]
                # else:
                #     hist["w_inp2exc"]["t"] = np.concatenate(
                #         [hist["w_inp2exc"]["t"], mon_state["t"][idx]]
                #     )
                #     hist["w_inp2exc"]["w"] = np.concatenate(
                #         [hist["w_inp2exc"]["w"], mon_state["w"][idx]]
                #     )

                cur_sim_time += duration + wait
                tot_iter += 1

        # Save weights and delays
        self._weights_inp2exc = synapses["inp2exc"].w
        self._weights_exc2inh = synapses["exc2inh"].w
        self._weights_inh2exc = synapses["inh2exc"].w
        self._delays_inp2exc = synapses["inp2exc"].delay

        # Check average response of each neuron
        for n in range(self._n_pool):
            for c in self._classes:
                firing_stats[n][c] /= tot_iter
                logging.info(f"Neuron {n} fired {firing_stats[n][c]:.2f} times for gesture {c}, on average.")

        # Decide association between neurons and classes
        for n in range(self._n_pool):
            c = max(firing_stats[n], key=firing_stats[n].get)
            self._readout[n] = c
        
        return hist

    def inference(
            self, firings_test_sample: pd.DataFrame
    ) -> str | None:
        """Test the SNN.

        Parameters
        ----------
        firings_test_sample : list of tuple of (DataFrame, int, float)
            Tuple containing the firings of MUs, the label of the gesture and the duration (in seconds).

        Returns
        -------
        str or None
            Predicted class label.
        """
        firings_test_x, firings_test_y, duration = firings_test_sample
        logging.info(f"Gesture {firings_test_y} - Duration: {duration:.2f} s")

        # Build fixed SNN
        neuron_groups, synapses, monitors = self._build_snn(plastic=False)
        snn = b2.Network(neuron_groups, synapses, monitors)

        # Set input spikes
        neuron_groups["inp"].set_spikes(
            firings_test_x["MU index"].values,
            firings_test_x["Firing time"].values * b2.second,
        )
        # Run simulation
        snn.run(duration * b2.second)

        # Majority voting based on the firing rate of each neuron
        spikes = np.array([np.count_nonzero(monitors["spikes_exc"].i == n) for n in range(self._n_pool)])
        firings_per_class = {c: 0 for c in self._classes}
        for n in range(self._n_pool):
            logging.info(f"Neuron {n} fired {spikes[n]} times.")
            if spikes[n] > 0:
                firings_per_class[self._readout[n]] += 1
        class_pred = max(firings_per_class, key=firings_per_class.get)
        # Check for ties
        if sum(1 for c, f in firings_per_class.items() if f == firings_per_class[class_pred]) != 1:
            class_pred = None
        logging.info(f"Class predicted: {class_pred}")

        return class_pred

    # noinspection PyTypeChecker
    def _build_snn(
            self, plastic: bool = True
    ) -> tuple[
        dict[str, b2.SpikeGeneratorGroup | b2.NeuronGroup],
        dict[str, b2.Synapses],
        dict[str, b2.SpikeMonitor]
    ]:
        """Build an instance of SNN from the given firings.

        Parameters
        ----------
        plastic : bool, default=True
            Whether to make the synapses plastic or fixed.

        Returns
        -------
        dict of {str, SpikeGeneratorGroup or NeuronGroup}
            Dictionary containing neuron groups.
        dict of {str, Synapses}
            Dictionary containing synapses.
        dict of {str, SpikeMonitor}
            Dictionary containing monitors.
        """
        # Define equations for neurons
        eqs_lif_exc = f"""
            dV/dt = (({self._v_rest_e} * mV - V) + (I_synE + I_synI) / nS) / ({self._tau_exc} * ms)
             : volt (unless refractory)
            I_synE = ge * nS * ({self._v_exc} * mV - V) : amp
            I_synI = gi * nS * ({self._v_inh} * mV - V) : amp
            dge/dt = -ge / ({self._tau_ge} * ms) : 1
            dgi/dt = -gi / ({self._tau_gi} * ms) : 1
        """
        eqs_reset_exc = f"V = {self._v_reset_e} * mV"
        eqs_thres_exc = f"V > {self._v_thres_e} * mV"
        
        eqs_lif_inh = f"""
            dV/dt = (({self._v_rest_i} * mV - V) + (I_synE + I_synI) / nS) / ({self._tau_inh} * ms)
             : volt (unless refractory)
            I_synE = ge * nS * ({self._v_exc} * mV - V) : amp
            I_synI = gi * nS * ({self._v_inh} * mV - V) : amp
            dge/dt = -ge / ({self._tau_ge} * ms) : 1
            dgi/dt = -gi / ({self._tau_gi} * ms) : 1
        """

        # Define layers:
        # 1. Input layer
        neurons_inp = b2.SpikeGeneratorGroup(self._n_inp, [0], [0] * b2.second)
        # 2. Excitatory layer
        neurons_exc = b2.NeuronGroup(
            N=self._n_pool,
            model=eqs_lif_exc,
            threshold=eqs_thres_exc,
            reset=eqs_reset_exc,
            refractory=self._refrac_e * b2.ms,
            method="euler",
        )
        neurons_exc.V = self._v_rest_e * b2.mV
        # 3. Inhibitory layer
        neurons_inh = b2.NeuronGroup(
            N=self._n_pool,
            model=eqs_lif_inh,
            threshold=f"V > {self._v_thres_i} * mV",
            reset=f"V = {self._v_reset_i} * mV",
            refractory=self._refrac_i * b2.ms,
            method="euler",
        )
        neurons_inh.V = self._v_rest_i * b2.mV

        # Define equations for synapses
        eqs_syn_exc = f"""
            w : 1
            dpre/dt = -pre / ({self._tau_pre} * ms) : 1 (event-driven)
            dpost/dt = -post / ({self._tau_post} * ms) : 1 (event-driven)
        """ if plastic else "w : 1"
        eqs_pre_exc = f"""
            ge += w
            pre += {self._max_pot * self._w_max}
            w = clip(w + post, 0., {self._w_max})
        """ if plastic else "ge += w"
        eqs_post_exc = f"""
            post -= {self._max_dep * self._w_max}
            w = clip(w + pre, 0., {self._w_max})
        """ if plastic else None

        # Define synapses
        # 1. From input to excitatory
        syn_inp2exc = b2.Synapses(
            source=neurons_inp,
            target=neurons_exc,
            model=eqs_syn_exc,
            on_pre=eqs_pre_exc,
            on_post=eqs_post_exc,
            method="linear",
        )
        syn_inp2exc.connect(True)  # all-to-all
        syn_inp2exc.w = f"rand() * {self._w_max}" if plastic else self._weights_inp2exc
        syn_inp2exc.delay = "rand() * 4 * ms" if plastic else self._delays_inp2exc
        # 2. From excitatory to inhibitory
        syn_exc2inh = b2.Synapses(
            source=neurons_exc,
            target=neurons_inh,
            model="w : 1",
            on_pre="ge += w",
            method="linear",
        )
        syn_exc2inh.connect(condition="i == j")  # one-to-one
        syn_exc2inh.w = f"rand() * {self._w_max}" if plastic else self._weights_exc2inh
        # 3. From inhibitory to excitatory
        syn_inh2exc = b2.Synapses(
            source=neurons_inh,
            target=neurons_exc,
            model="w : 1",
            on_pre="gi += w",
            method="linear",
        )
        syn_inh2exc.connect(condition="i != j")  # one-to-all
        syn_inh2exc.w = f"rand() * {self._w_max}" if plastic else self._weights_inh2exc

        neuron_groups = {
            "inp": neurons_inp,
            "exc": neurons_exc,
            "inh": neurons_inh
        }
        synapses = {
            "inp2exc": syn_inp2exc,
            "exc2inh": syn_exc2inh,
            "inh2exc": syn_inh2exc
        }
        monitors = {
            "spikes_inp": b2.SpikeMonitor(neurons_inp),
            "spikes_exc": b2.SpikeMonitor(neurons_exc),
            "spikes_inh": b2.SpikeMonitor(neurons_inh),
            "V_exc": b2.StateMonitor(neurons_exc, "V", record=True),
            "V_inh": b2.StateMonitor(neurons_inh, "V", record=True),
            # "w_inp2exc": b2.StateMonitor(syn_inp2exc, "w", record=True)
        }

        return neuron_groups, synapses, monitors
