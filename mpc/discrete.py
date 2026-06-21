#!/usr/bin/env python3

"""Discrete-time system models for Model Predictive Control.

This module provides abstract base classes and concrete
implementations for discrete-time systems. These systems define the
state transition, control input, and output relationships used by the
MPC controller.

Classes:
    DiscreteSystem: abstract base class for discrete-time systems.
    LtiSystem: discrete linear time-invariant system.
    NonlinearSystem: generalized discrete non-linear system.
    HomogeneousSystem: discrete homogeneous non-linear system.
"""

from __future__ import annotations

import abc
from typing import Optional, Callable, override, final
import numpy as np


__all__ = ('LtiSystem', 'AtiSystem',
           'HomogeneousSystem', 'NonlinearSystem')


class Discrete(abc.ABC):
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
    def _linearize_transition(self,
                              state: Optional[np.ndarray] = None,
                              control: Optional[np.ndarray] = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the state transition function.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the transition
                matrix A, control matrix B, and disturbance d.

        """
        ...

    @abc.abstractmethod
    def _linearize_output(self,
                          state: Optional[np.ndarray] = None,
                          ) -> np.ndarray:
        """Linearize the output function.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output matrix C.
        """
        ...

    def linearize(self,
                  state: Optional[np.ndarray] = None,
                  control: Optional[np.ndarray] = None
                  ) -> AffineTimeInvariant:
        """Return the lti system based on given states.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            LtiSystem: the linearized system.
        """
        transition_matrix, control_matrix, disturbance_vector = \
            self._linearize_transition(state, control)
        output_matrix = self._linearize_output(state)
        return AtiSystem(transition_matrix,
                         control_matrix,
                         disturbance_vector,
                         output_matrix)

    def _get_state_one_step(self,
                            state: np.ndarray,
                            control: np.ndarray) -> np.ndarray:
        """Evaluate the next state vector.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            np.ndarray: the state sequence of shape (n_state,).
        """
        a, b, d = self._linearize_transition(state, control)
        return a @ state + b @ control + d

    def get_state(self,
                  initial_state: np.ndarray,
                  controls: np.ndarray
                  ) -> np.ndarray:
        """Evaluate the states based on given input.

        Args:
            initial_state: the initial state of shape (n_state, ).
            controls: the control input of shape (n_steps, n_control).

        Returns:
            np.ndarray: the state sequence of shape (n_steps, n_state).
        """
        n = controls.shape[0]
        xs = np.zeros([n, self.n_state])
        state = initial_state
        for (i, control) in enumerate(controls):
            next_state = self._get_state_one_step(state, control)
            xs[i] = next_state
            state = next_state
        return xs

    def _get_output_one_step(self,
                             state: np.ndarray
                             ) -> np.ndarray:
        """Evaluate the output vector for a single step.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output vector of shape (n_output, ).
        """
        c = self._linearize_output(state)
        return c @ state

    def get_output(self,
                   states: np.ndarray
                   ) -> np.ndarray:
        """Evaluate the outputs based on given states.

        Args:
            states: the state sequence of shape (n_steps, n_state),
                    typically obtained from get_state().

        Returns:
            np.ndarray: the output sequence of shape (n_steps, n_output).
        """
        n = states.shape[0]
        ys = np.zeros([n, self.n_output])
        for (i, state) in enumerate(states):
            ys[i] = self._get_output_one_step(state)
        return ys


class AffineTimeInvariant(Discrete):
    """Base class (trait) for affine time invariant (ATI) systems.

    x[n+1] = A @ x[n] + B @ u[n] + d
    y[n] = C @ x[n]
    """

    @abc.abstractproperty
    def transition_matrix(self) -> np.ndarray:
        """Return the state transition matrix A."""
        ...

    @abc.abstractproperty
    def control_matrix(self) -> np.ndarray:
        """Return the control matrix B."""
        ...

    @abc.abstractproperty
    def disturbance_vector(self) -> np.ndarray:
        """Return the disturbance vector d."""
        ...

    @abc.abstractproperty
    def output_matrix(self) -> np.ndarray:
        """Return the output matrix C."""
        ...

    @override
    def _linearize_transition(self,
                              state: Optional[np.ndarray] = None,
                              control: Optional[np.ndarray] = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linearize the state transition function.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the transition
                matrix A, control matrix B, and disturbance d.

        """
        return (self.transition_matrix,
                self.control_matrix,
                self.disturbance_vector)

    @override
    def _linearize_output(self,
                          state: Optional[np.ndarray] = None,
                          ) -> np.ndarray:
        """Linearize the output function.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output matrix C.
        """
        return self.output_matrix


@final
class AtiSystem(AffineTimeInvariant):
    """Affine time invariant system.

    x[n+1] = A @ x[n] + B @ u[n] + d
    y[n] = C @ x[n]
    """

    def __init__(self,
                 transition_matrix: np.ndarray,
                 control_matrix: np.ndarray,
                 disturbance_vector: np.ndarray,
                 output_matrix: np.ndarray
                 ):
        """Initialize the ATI system.

        Args:
            transition_matrix: the state transition matrix A of shape
                (n_state, n_state).
            control_matrix: the control matrix B of shape
                (n_state, n_control).
            disturbance_vector: the disturbance vector d of shape
                (n_state,).
            output_matrix: the output matrix C of shape
                (n_output, n_state).
        """
        self._a = np.asarray(transition_matrix)
        self._b = np.asarray(control_matrix)
        self._d = np.asarray(disturbance_vector)
        self._c = np.asarray(output_matrix)
        self._n_state = self._a.shape[0]
        self._n_control = self._b.shape[1]
        self._n_output = self._c.shape[0]

    @override
    @property
    def n_state(self) -> int:
        """Dimension of state vector."""
        return self._n_state

    @override
    @property
    def n_control(self) -> int:
        """Dimension of control vector."""
        return self._n_control

    @override
    @property
    def n_output(self) -> int:
        """Dimension of output vector."""
        return self._n_output

    @override
    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the state transition matrix A."""
        return self._a

    @override
    @property
    def control_matrix(self) -> np.ndarray:
        """Return the control matrix B."""
        return self._b

    @override
    @property
    def disturbance_vector(self) -> np.ndarray:
        """Return the disturbance vector d."""
        return self._d

    @override
    @property
    def output_matrix(self) -> np.ndarray:
        """Return the output matrix C."""
        return self._c


@final
class LtiSystem(AffineTimeInvariant):
    """
    Discrete Linear Time-Invariant System.

    Equation:
        x[n+1] = A @ x[n] + B @ u[n]
        y[n] = C @ x[n]
    """

    def __init__(self,
                 transition_matrix: np.ndarray,
                 control_matrix: np.ndarray,
                 output_matrix: np.ndarray
                 ):
        """Initialize the LTI system.

        Args:
            transition_matrix: the state transition matrix A of shape
                (n_state, n_state).
            control_matrix: the control matrix B of shape
                (n_state, n_control).
            output_matrix: the output matrix C of shape
                (n_output, n_state).
        """
        self._a = np.asarray(transition_matrix)
        self._b = np.asarray(control_matrix)
        self._c = np.asarray(output_matrix)
        self._n_state = self._a.shape[0]
        self._n_control = self._b.shape[1]
        self._n_output = self._c.shape[0]

    @override
    @property
    def n_state(self) -> int:
        """Dimension of state vector."""
        return self._n_state

    @override
    @property
    def n_control(self) -> int:
        """Dimension of control vector."""
        return self._n_control

    @override
    @property
    def n_output(self) -> int:
        """Dimension of output vector."""
        return self._n_output

    @override
    @property
    def transition_matrix(self) -> np.ndarray:
        """Return the state transition matrix A."""
        return self._a

    @override
    @property
    def control_matrix(self) -> np.ndarray:
        """Return the control matrix B."""
        return self._b

    @override
    @property
    def disturbance_vector(self) -> np.ndarray:
        """Return the disturbance vector d."""
        return np.zeros([self.n_state])

    @override
    @property
    def output_matrix(self) -> np.ndarray:
        """Return the output matrix C."""
        return self._c


@final
class HomogeneousSystem(Discrete):
    """
    Discrete non-linear homogeneous system.

    Equation:
        x[n+1] = A(x[n], u[n]) @ x[n] + B(x[n], u[n]) @ u[n]
        y[n] = C(x[n]) @ x[n]
    """

    def __init__(
            self,
            n_state: int,
            n_control: int,
            n_output: int,
            transition_matrix: Callable[[np.ndarray, np.ndarray],
                                        np.ndarray],
            control_matrix: Callable[[np.ndarray, np.ndarray],
                                     np.ndarray],
            output_matrix: Callable[[np.ndarray],
                                    np.ndarray]):
        """Initialize the discrete non-linear system with disturbance.

        Args:
            n_state: dimension of the state vector.
            n_control: dimension of the control vector.
            n_output: dimension of the output vector.
            transition_matrix: callable that returns the state transition
                matrix A. Signature: A(state, control) -> np.ndarray of
                shape (n_state, n_state).
            control_matrix: callable that returns the control matrix B.
                Signature: B(state, control) -> np.ndarray of shape
                (n_state, n_control).
            output_matrix: callable that returns the output matrix C.
                Signature: C(state) -> np.ndarray of shape
                (n_output, n_state).
        """
        self._n_state = n_state
        self._n_control = n_control
        self._n_output = n_output
        self._a = transition_matrix
        self._b = control_matrix
        self._c = output_matrix

    @override
    @property
    def n_state(self) -> int:
        """Dimension of state vector."""
        return self._n_state

    @override
    @property
    def n_control(self) -> int:
        """Dimension of control vector."""
        return self._n_control

    @override
    @property
    def n_output(self) -> int:
        """Dimension of output vector."""
        return self._n_output

    @override
    def _linearize_transition(self,
                              state: Optional[np.ndarray] = None,
                              control: Optional[np.ndarray] = None
                              ) -> tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray]:
        """Linearize the state transition function.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the transition
                matrix A, control matrix B, and disturbance d.
        """
        if state is None:
            raise ValueError('state can not be None')
        if control is None:
            raise ValueError('control can not be None')
        return (self._a(state, control),
                self._b(state, control),
                np.zeros([self.n_state]))

    @override
    def _linearize_output(self,
                          state: Optional[np.ndarray] = None,
                          ) -> np.ndarray:
        """Linearize the output function.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output matrix C.
        """
        if state is None:
            raise ValueError('state can not be None')
        return self._c(state)


@final
class NonlinearSystem(Discrete):
    """
    Discrete non-linear System.

    Equation:
        x[n+1] = A(x[n], u[n]) @ x[n] + B(x[n], u[n]) @ u[n] + d(x[n], u[n])
        y[n] = C(x[n]) @ x[n]
    """

    def __init__(
            self,
            n_state: int,
            n_control: int,
            n_output: int,
            transition_matrix: Callable[[np.ndarray, np.ndarray],
                                        np.ndarray],
            control_matrix: Callable[[np.ndarray, np.ndarray],
                                     np.ndarray],
            disturbance_vector: Callable[[np.ndarray, np.ndarray],
                                         np.ndarray],
            output_matrix: Callable[[np.ndarray],
                                    np.ndarray]):
        """Initialize the discrete non-linear system with disturbance.

        Args:
            n_state: dimension of the state vector.
            n_control: dimension of the control vector.
            n_output: dimension of the output vector.
            transition_matrix: callable that returns the state transition
                matrix A. Signature: A(state, control) -> np.ndarray of
                shape (n_state, n_state).
            control_matrix: callable that returns the control matrix B.
                Signature: B(state, control) -> np.ndarray of shape
                (n_state, n_control).
            disturbance_vector: callable that returns the disturbance
                vector vector d. Signature: d(state, control) -> np.ndarray
                of shape (n_state,).
            output_matrix: callable that returns the output matrix C.
                Signature: C(state) -> np.ndarray of shape
                (n_output, n_state).
        """
        self._n_state = n_state
        self._n_control = n_control
        self._n_output = n_output
        self._a = transition_matrix
        self._b = control_matrix
        self._d = disturbance_vector
        self._c = output_matrix

    @override
    @property
    def n_state(self) -> int:
        """Dimension of state vector."""
        return self._n_state

    @override
    @property
    def n_control(self) -> int:
        """Dimension of control vector."""
        return self._n_control

    @override
    @property
    def n_output(self) -> int:
        """Dimension of output vector."""
        return self._n_output

    @override
    def _linearize_transition(self,
                              state: Optional[np.ndarray] = None,
                              control: Optional[np.ndarray] = None
                              ) -> tuple[np.ndarray,
                                         np.ndarray,
                                         np.ndarray]:
        """Linearize the state transition function.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the transition
                matrix A, control matrix B, and disturbance d.
        """
        if state is None:
            raise ValueError('state can not be None')
        if control is None:
            raise ValueError('control can not be None')
        return (self._a(state, control),
                self._b(state, control),
                self._d(state, control))

    @override
    def _linearize_output(self,
                          state: Optional[np.ndarray] = None,
                          ) -> np.ndarray:
        """Linearize the output function.

        Args:
            state: current state vector of shape (n_state, ).

        Returns:
            np.ndarray: the output matrix C.
        """
        if state is None:
            raise ValueError('state can not be None')
        return self._c(state)
