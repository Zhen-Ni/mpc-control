#!/usr/bin/env python3

"""Implementation of the Kalman filter / Extended Kalman Filter."""


from __future__ import annotations

from typing import Optional
import numpy as np

from .discrete import Discrete, AffineTimeInvariant


__all__ = ['Ekf', 'Ukf']


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

    Attributes:
        system: The discrete-time system model. Can be linear or nonlinear.
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

        Args:
            u: The input variable (control vector).
            q: The process noise uncertainty of the current timestep.
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

        Args:
            z: The measured output of the system.
            r: The measurement uncertainty of the current timestep.
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


class Ukf:
    """Implementation of the Unscented Kalman Filter (UKF).

    The implementation uses the unscented transform to handle
    nonlinear system models provided by a `Discrete` object.
    Unlike the EKF, it does not require Jacobian linearization and
    instead propagates a set of sigma points through the system
    dynamics.

    The system model must be passed during instantiation:
        kalman_filter = Ukf(system, x0, P0, alpha, beta, kappa)

    The state vector and its uncertainty matrix can be accessed by
    attribute `x` and `P`.

    After initialization, the filter is fully prepared for making
    predictions for the next timestep, or updating its state by new
    measurement:
        kalman_filter.predict(u, Q)
        kalman_filter.update(z, R)

    Attributes:
        system: The discrete-time system model. Can be linear or nonlinear.
        alpha: Spread of the sigma points. Typically 1e-3.
        beta: Incorporation of prior knowledge. 2 is optimal for Gaussian.
        kappa: Secondary scaling parameter. Typically 0.
    """

    def __init__(self, system: Discrete,
                 x0: Optional[np.ndarray] = None,
                 p0: Optional[np.ndarray] = None,
                 alpha: float = 1e-3,
                 beta: float = 2.0,
                 kappa: float = 0.0):
        """Initialize the Unscented Kalman filter.

        Args:
            system: A discrete-time system model (`Discrete`).
            x0: Initial state vector. Defaults to a zero vector.
            p0: Initial state covariance matrix. Defaults to an
                identity matrix.
            alpha: Spread of the sigma points.
            beta: Incorporation of prior knowledge.
            kappa: Secondary scaling parameter.
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

        # UKF parameters
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._n_state = n_state
        self._lambda = alpha**2 * (n_state + kappa) - n_state

        # Calculate weights
        self._wm = np.zeros(2 * n_state + 1)
        self._wc = np.zeros(2 * n_state + 1)
        self._wm[0] = self._lambda / (n_state + self._lambda)
        self._wc[0] = self._wm[0] + (1 - alpha**2 + beta)
        for i in range(1, 2 * n_state + 1):
            self._wm[i] = 1.0 / (2 * (n_state + self._lambda))
            self._wc[i] = self._wm[i]

        # Cache for Kalman gain
        self._k: Optional[np.ndarray] = None
        # Cache for predicted sigma points to be used in update
        self._sigmas_pred: Optional[np.ndarray] = None

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

    def _generate_sigmas(self) -> np.ndarray:
        """Generate sigma points based on current state and covariance.

        Returns:
            np.ndarray: Sigma points of shape (2 * n_state + 1, n_state).
        """
        # np.linalg.cholesky returns lower triangular L such that L @ L.T = P
        L = np.linalg.cholesky(self._p)

        sigmas = np.zeros([2 * self._n_state + 1, self._n_state])
        sigmas[0] = self._x
        gamma = (self._n_state + self._lambda) ** .5
        for i in range(self._n_state):
            sigmas[i + 1] = self._x + gamma * L[:, i]
            sigmas[self._n_state + i + 1] = self._x - gamma * L[:, i]
        return sigmas

    def predict(self, u: np.ndarray, q: np.ndarray) -> None:
        """Predict the system state at the next step.

        Both the state vector and its uncertainty at the
        next timestep are predicted by given input
        vector `u` and its uncertainty matrix `q`.

        Args:
            u: The input variable (control vector).
            q: The process noise uncertainty of the current timestep.
        """
        sigmas = self._generate_sigmas()

        # Propagate sigma points through the state transition function
        sigmas_pred = np.zeros_like(sigmas)
        for i in range(len(sigmas)):
            sigmas_pred[i] = self._system._get_state_one_step(sigmas[i], u)

        # Calculate predicted state mean
        x_pred = self._wm @ sigmas_pred

        # Calculate predicted state covariance
        p_pred = np.zeros_like(self._p)
        for i in range(len(sigmas)):
            diff = sigmas_pred[i] - x_pred
            p_pred += self._wc[i] * np.outer(diff, diff)
        p_pred += q

        self._x = x_pred
        self._p = p_pred
        self._sigmas_pred = sigmas_pred

    def update(self, z: np.ndarray, r: np.ndarray) -> None:
        """Update the system state by measurement.

        Both the state vector and its uncertainty at the
        next timestep are updated by given measurement
        vector `z` and its uncertainty matrix `r`.

        Args:
            z: The measured output of the system.
            r: The measurement uncertainty of the current timestep.
        """
        if self._sigmas_pred is None:
            raise RuntimeError('Should call `predict` before updating.')

        n_output = self._system.n_output

        # Propagate sigma points through the observation function
        sigmas_obs = np.zeros([len(self._sigmas_pred), n_output])
        for i in range(len(self._sigmas_pred)):
            sigmas_obs[i] = self._system._get_output_one_step(
                self._sigmas_pred[i])

        # Calculate predicted measurement mean
        z_pred = self._wm @ sigmas_obs

        # Calculate measurement covariance S
        S = np.zeros([n_output, n_output])
        for i in range(len(sigmas_obs)):
            diff = sigmas_obs[i] - z_pred
            S += self._wc[i] * np.outer(diff, diff)
        S += r

        # Calculate cross covariance C
        C = np.zeros((self._n_state, n_output))
        for i in range(len(sigmas_obs)):
            diff_x = self._sigmas_pred[i] - self._x
            diff_z = sigmas_obs[i] - z_pred
            C += self._wc[i] * np.outer(diff_x, diff_z)

        # Calculate Kalman gain k = C @ inv(S)
        # Equivalent to k.T = solve(S, C.T)
        k = np.linalg.solve(S, C.T).T

        # Update state and covariance
        z_arr = np.asarray(z)
        x_n_stateew = self._x + k @ (z_arr - z_pred)
        p_n_stateew = self._p - k @ S @ k.T

        # Update caches
        self._k = k
        self._x = x_n_stateew
        self._p = p_n_stateew
        self._sigmas_pred = None
