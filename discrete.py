#!/usr/bin/env python3

from __future__ import annotations

import abc
from typing import Optional, Self
import numpy as np


class DiscreteSystem(abc.ABC):
    """Abstract base class for discrete-time systems.

    The system can be written as:
    x[n+1] = f(x[n], u[n])
    y[n] = g(x[n])
    """

    @abc.abstractproperty
    def n_state(self) -> int:
        """Dimension of state vector."""
        ...

    @abc.abstractproperty
    def n_control(self) -> int:
        """Dimension of control vector."""
        ...

    @abc.abstractproperty
    def n_output(self) -> int:
        """Dimension of output vector."""
        ...

    @abc.abstractmethod
    def linearize(self,
                  state: Optional[np.ndarray] = None,
                  control: Optional[np.ndarray] = None
                  ) -> LtiSystem:
        """Returns the lti system based on given states."""
        ...

    @abc.abstractmethod
    def _get_state_one_step(self,
                            state: np.ndarray,
                            control: np.ndarray) -> np.ndarray:
        """Evaluate the next state vector.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (ncontrol, ).
        """
        ...

    def get_state(self,
                  initial_state: np.ndarray,
                  controls: np.ndarray
                  ) -> np.ndarray:
        """Evaluate the states based on given input.

        Args:
            initial_state: the initial state of shape (n_state, ).
            controls: the control input of shape (nsteps, ncontrol).

        Returns:
            np.ndarray: the state sequence of shape (nsteps, n_state).
        """
        n = controls.shape[0]
        xs = np.zeros([n, self.n_state])
        state = initial_state
        for (i, control) in enumerate(controls):
            next_state = self._get_state_one_step(state, control)
            xs[i] = next_state
            state = next_state
        return xs

    @abc.abstractmethod
    def _get_output_one_step(self,
                             state: np.ndarray
                             ) -> np.ndarray:
        """Evaluate the output vector for a single step.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output vector of shape (n_output, ).
        """
        ...

    def get_output(self,
                   states: np.ndarray
                   ) -> np.ndarray:
        """Evaluate the outputs based on given states.

        Args:
            states: the state sequence of shape (nsteps, n_state), 
                    typically obtained from get_state().

        Returns:
            np.ndarray: the output sequence of shape (nsteps, n_output).
        """
        n = states.shape[0]
        ys = np.zeros([n, self.n_output])
        for (i, state) in enumerate(states):
            ys[i] = self._get_output_one_step(state)
        return ys


class LtiSystem(DiscreteSystem):
    """
    Discrete Linear Time-Invariant System.

    Equation: x[n+1] = A * x[n] + B * u[n]
    y[n] = C * x[n]
    """

    def __init__(self,
                 transition_matrix: np.ndarray,
                 control_matrix: np.ndarray,
                 output_matrix: np.ndarray
                 ):
        self._a = transition_matrix
        self._b = control_matrix
        self._c = output_matrix
        self._n_state = self._a.shape[0]
        self._n_control = self._b.shape[1]
        self._n_output = self._c.shape[0]

        # Input validation
        if (self._a.shape[1] != self._n_state or
            self._b.shape[0] != self._n_state or
                self._c.shape[1] != self._n_state):
            raise ValueError(
                "Dimension of "
                "state_transiton matrix "
                f"[{self._a.shape[0]}, {self._a.shape[1]}], "
                "control_matrix "
                f"[{self._b.shape[0]}, {self._b.shape[1]}], "
                "output_matrix "
                f"[{self._c.shape[0]}, {self._c.shape[1]}], "
                "incorrect")

    @property
    def n_state(self) -> int:
        return self._n_state

    @property
    def n_control(self) -> int:
        return self._n_control

    @property
    def n_output(self) -> int:
        return self._n_output

    def linearize(self,
                  _state: Optional[np.ndarray] = None,
                  _control: Optional[np.ndarray] = None
                  ) -> Self:
        return self

    def get_transition_matrix(self) -> np.ndarray:
        """Returns the state transition matrix A."""
        return self._a

    def get_control_matrix(self) -> np.ndarray:
        """Returns the control matrix B."""
        return self._b

    def get_output_matrix(self) -> np.ndarray:
        """Returns the output matrix C."""
        return self._c

    def _get_state_one_step(self,
                            state: np.ndarray,
                            control: np.ndarray) -> np.ndarray:
        """Evaluate the next state vector.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (ncontrol, ).

        Returns:
            np.ndarray: the next state vector of shape (n_state, ).
        """
        return self._a @ state + self._b @ control

    def _get_output_one_step(self,
                             state: np.ndarray
                             ) -> np.ndarray:
        """Evaluate the output vector for a single step.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output vector of shape (n_output, ).
        """
        return self._c @ state
