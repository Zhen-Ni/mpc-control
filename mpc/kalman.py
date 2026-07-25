#!/usr/bin/env python3

"""Implementation of the Kalman filter / Extended Kalman Filter."""


from __future__ import annotations

from typing import Optional
import numpy as np

from .discrete import Discrete, AffineTimeInvariant


__all__ = ['Ekf']


class Ekf:
    """Implementation of a Kalman filter / Extended Kalman Filter.

    The implementation of the code is based on Ref [1]_. The system
    model is provided by a `Discrete` object, which supplies the
    state transition, input transition, and observation matrices.
    For nonlinear systems, the `Discrete` object is linearized at
    each timestep, making this an Extended Kalman Filter (EKF).

    The system model must be passed during instantiation:
        kalman_filter = Kalman(system, x0, P0)

    The state vector and its uncertainty matrix can be accessed by
    attribute `x` and `P`.

    After initialization, the filter is fully prepared for making
    predictions for the next timestep, or updating its state by new
    measurement:
        kalman_filter.predict(u, Q)
        kalman_filter.update(z, R)
    Users can refer to the docstring of these two member funcitons
    for the usage.

    Parameters
    ----------
    system: Discrete
        The discrete-time system model. Can be linear or nonlinear.

    Notes
    -----
    It is the user's responsibility to make sure the system model
    and initial x and P are correctly set.

    References
    ----------
    .. [1] kalmanfilter.net
    """

    def __init__(self, system: Discrete,
                 x0: Optional[np.ndarray] = None,
                 p0: Optional[np.ndarray] = None):
        """Initialize the Kalman filter.

        Args:
            system: A discrete-time system model (`Discrete`).
            x0: Initial state vector. Defaults to a zero vector.
            p0: Initial state covariance matrix. Defaults to an
                identity matrix.
        """
        self._system = system
        n_state = self._system.n_state

        if x0 is None:
            self._x = np.zeros(n_state)
        else:
            x0_arr = np.asarray(x0)
            if x0_arr.shape != (n_state,):
                raise ValueError(
                    f"x0 must have shape ({n_state},), "
                    f"got {x0_arr.shape}")
            self._x = x0_arr

        if p0 is None:
            self._p = np.eye(n_state)
        else:
            p0_arr = np.asarray(p0)
            if p0_arr.shape != (n_state, n_state):
                raise ValueError(
                    f"p0 must have shape ({n_state}, {n_state}), "
                    f"got {p0_arr.shape}")
            self._p = p0_arr

        # Cache for Kalman gain
        self._k: Optional[np.ndarray] = None
        # Cached linearlized system
        self._linearized: Optional[AffineTimeInvariant] = None

    @property
    def system(self) -> Discrete:
        """Access to the system model."""
        return self._system

    @property
    def x(self) -> np.ndarray:
        """Access to the system state."""
        return self._x

    @property
    def p(self) -> np.ndarray:
        """Access to the state covariance matrix."""
        return self._p

    @property
    def k(self) -> np.ndarray | None:
        """Access to Kalman gain."""
        return self._k

    def predict(self, u: np.ndarray, q: np.ndarray) -> None:
        """Predict the system state at the next step.

        Both the state vector and its uncertainty at the
        next timestep are predicted by given input
        vector `u` and its uncertainty matrix `q`.

        Note that its the user's responsibility to make
        sure the dimensions of the input arguments are
        correct.

        Parameters
        ----------
        u : np.ndarray
            The input variable (control vector).
        q : np.ndarray
            The process noise uncertainty of the current timestep.
        """
        # Update the system
        self._linearized = self._system.linearize(self._x, u)
        a = self._linearized.transition_matrix
        b = self._linearized.control_matrix
        d = self._linearized.disturbance_vector
        x = a @ self._x + b @ u + d
        P = a @ self._p @ a.T + q
        self._x = x
        self._p = P

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        """Update the system state by measurement.

        Both the state vector and its uncertainty at the
        next timestep are updated by given measurement
        vector `z` and its uncertainty matrix `r`.

        Note that its the user's responsibility to make
        sure the dimensions of the input arguments are
        correct.

        Parameters
        ----------
        z : np.ndarray
            The measured output of the system.
        r : np.ndarray
            The measurement uncertainty of the current timestep.
        """
        if self._linearized is None:
            raise RuntimeError('Should call `predict` before updating.')
        c = self._linearized.output_matrix
        # Calculate innovation covariance `S`.
        S = c @ self._p @ c.T + r
        # Calculate Kalman gain `k`.
        # k = p @ C.T @ inv(S) is equivalent to k.T = solve(S, C @ p)
        k = np.linalg.solve(S, c @ self._p).T
        # Calculate the updated state vector `x`.
        x = self._x + k @ (np.asarray(z) - c @ self._x)
        # Calculate the updated uncertainty `p` for the state vector.
        T = np.eye(k.shape[0]) - k @ c
        p = T @ self._p @ T.T + k @ r @ k.T
        # Update self.k, self.x and self.p.
        self._k = k
        self._x = x
        self._p = p
