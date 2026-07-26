#!/usr/bin/env python3

"""Simulation environment for discrete systems.

This module provides simulation environments (plants) that wrap a
discrete-time system. These plants hold the current state of the
system and step through simulation time given a sequence of control
inputs.

Classes:
    Plant: basic simulation environment for a discrete system.
    LoggedPlant: simulation environment with state/input/output logging.
"""

from typing import Optional
from collections import deque
import numpy as np

from .discrete import Discrete


__all__ = ('Plant', 'LoggedPlant')


class Plant:
    """Basic simulation environment for a discrete system.

    This class maintains the current state of the system and updates
    it based on the control inputs.
    """

    def __init__(self,
                 system: Discrete,
                 initial_state: Optional[np.ndarray] = None):
        """Initialize the plant.

        Args:
            system: the discrete-time system to be simulated.
            initial_state: the initial state vector of shape
                (n_state,). If None, the state is initialized to zeros.
        """
        self._system = system
        if initial_state is None:
            self._state = np.zeros(self._system.n_state)
        else:
            initial_state = np.asarray(initial_state, dtype=float)
            if initial_state.shape != (self._system.n_state,):
                raise ValueError('shape of initial state and '
                                 'system state not match')
            self._state = initial_state

    def step(self, control_input: np.ndarray) -> np.ndarray:
        """Step the simulation forward by one time step.

        Updates the internal state based on the control input and
        computes the corresponding output.

        Args:
            control_input: the control input vector of shape
                (n_control,).

        Returns:
            np.ndarray: the output vector of shape (n_output,).
        """
        x = self._system._get_state_one_step(self._state, control_input)
        y = self._system._get_output_one_step(x)
        self._state = x
        return y


class LoggedPlant(Plant):
    """Simulation environment with logging capabilities.

    This class extends `Plant` by recording the history of inputs,
    states, and outputs using a fixed-length deque.
    """

    def __init__(self,
                 system: Discrete,
                 initial_state: Optional[np.ndarray] = None,
                 maxlen: Optional[int] = None):
        """Initialize the logged plant.

        Args:
            system: the discrete-time system to be simulated.
            initial_state: the initial state vector of shape
                (n_state,). If None, the state is initialized to zeros.
            maxlen: maximum number of time steps to keep in the history.
                If None, the history grows unbounded.
        """
        super().__init__(system, initial_state)
        self._input_deque: deque[np.ndarray] = deque(maxlen=maxlen)
        self._state_deque: deque[np.ndarray] = deque(maxlen=maxlen)
        self._output_deque: deque[np.ndarray] = deque(maxlen=maxlen)

    def step(self, control_input: np.ndarray) -> np.ndarray:
        """Step the simulation forward by one time step.

        Updates the internal state, computes the corresponding output,
        and logs the input, state, and output to their respective
        deques.

        Args:
            control_input: the control input vector of shape
                (n_control,).

        Returns:
            np.ndarray: the output vector of shape (n_output,).
        """
        output = super().step(control_input)
        self._input_deque.append(control_input)
        self._state_deque.append(self._state)
        self._output_deque.append(output)
        return output

    def get_input_history(self):
        """Return the history of control inputs.

        Returns:
            np.ndarray: the control input history of shape
                (n_steps, n_control).
        """
        return np.asarray(self._input_deque)

    def get_state_history(self):
        """Return the history of states.

        Returns:
            np.ndarray: the state history of shape
                (n_steps, n_state).
        """
        return np.asarray(self._state_deque)

    def get_output_history(self):
        """Return the history of outputs.

        Returns:
            np.ndarray: the output history of shape
                (n_steps, n_output).
        """
        return np.asarray(self._output_deque)
