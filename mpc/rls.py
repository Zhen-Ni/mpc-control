#!/usr/bin/env python3

import abc
from typing import Optional, TypeVar, Generic
import numpy as np
from scipy import linalg
from . import discrete


_T_system = TypeVar('_T_system', bound=discrete.Discrete)


class Rls(abc.ABC, Generic[_T_system]):
    r"""Solver class for Recursive Least Squares (RLS).

    This class provides the RLS algorithm for online parameter
    identification. It assumes the system can be formulated such that
    the error is linear with respect to the parameters.

    Mathematical Derivation:
        1. Identification model offset:
           e = y - \hat{y}

        2. Cost function:
           J = \sum_{i=0}^{t} \lambda^{t-i} \|e(i)\|^2
           where \lambda is the forgetting factor.

        3. RLS Update Equations:
           K = P \Phi (\lambda I + \Phi^T P \Phi)^{-1}
           \theta_{new} = \theta + K e
           P_{new} = (P - K \Phi^T P) / \lambda

    where y is the true measurement and \hat{y} is the estimated output.
    n_parameter is the dimension of the parameter vector \theta,
    n_error is the dimension of the error vector e.
    P is the covariance matrix of shape (n_parameter, n_parameter).
    \Phi is the coefficient matrix of shape (n_parameter, n_error).

    """

    @abc.abstractmethod
    def get_parameter(self) -> np.ndarray:
        """Return the current parameter vector theta.

        Returns:
            np.ndarray: The parameter vector of shape (n_parameter, 1).
        """
        ...

    @abc.abstractmethod
    def set_parameter(self, parameters: np.ndarray) -> None:
        """Set the parameter vector theta.

        Args:
            parameters: The new parameter vector of shape (n_parameter, 1).
        """
        ...

    @abc.abstractmethod
    def get_coefficient(self,
                        state: Optional[np.ndarray] = None,
                        control: Optional[np.ndarray] = None
                        ) -> np.ndarray:
        """Return the current coefficient matrix Phi.

        Args:
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).

        Returns:
            np.ndarray: The coefficient matrix of shape (n_parameter, n_error).
        """
        ...

    def __init__(self, system: _T_system, delta: float):
        """Initialize the RLS solver.

        Args:
            system: The discrete system to be identified.
            delta: The initial scaling factor for the covariance matrix P.
                P will be initialized as `delta * I`, where I is the
                identity matrix of shape (n_parameter, n_parameter). A
                larger value makes the algorithm more responsive to
                initial errors, leading to faster initial convergence
                but potentially more oscillation. A smaller value
                results in slower convergence and relies more heavily
                on the initial parameter guess. Typical values are
                large (e.g., 100, 1000) when the initial guess is
                poor.

        """
        self._system = system
        self._delta = delta
        self._p: Optional[np.ndarray] = None

        self._parameter_cache: Optional[np.ndarray] = None

    @property
    def system(self) -> _T_system:
        """The target system instance to be identified."""
        return self._system

    def step(self,
             error: np.ndarray,
             forgetting_factor: float,
             state: Optional[np.ndarray] = None,
             control: Optional[np.ndarray] = None
             ):
        r"""Update the parameter estimate using the RLS algorithm.

        Args:
            error: The estimation offset vector of shape (n_error, 1).
                Must be defined as (y - \hat{y}), i.e., the true
                measurement minus the estimated output.
            forgetting_factor: The forgetting factor (lambda) that
                weights recent data more heavily. Should be in (0, 1).
                A value of 1.0 means no forgetting, treating all past
                data equally. However, using exactly 1.0 is not
                recommended as the covariance matrix P will decay
                towards zero over time, causing numerical instability
                and a frozen gain matrix. A value slightly less than
                1.0 exponentially discounts older data, allowing the
                algorithm to track slowly varying parameters and
                maintaining a non-zero P matrix. Smaller values yield
                faster tracking but increase sensitivity to noise.
                Typical values range from 0.95 to 0.999.
            state: current state vector of shape (n_state, ).
            control: current control input of shape (n_control, ).
        """
        if self._parameter_cache is None:
            self._parameter_cache = self.get_parameter().copy()
        theta = self._parameter_cache
        phi = self.get_coefficient(state, control)
        p = np.eye(len(theta)) * self._delta if self._p is None else self._p

        # 1. Calculate the gain matrix K (phi dim: n x m, error dim: m x 1)
        m = phi.shape[1]
        P_phi = p @ phi
        # The matrix S = lambda*I + Phi^T*P*Phi is symmetric positive
        # definite.  Using solve with assume_a='pos' exploits this.
        # This is equivalent to: ```K = P_phi @
        # np.linalg.inv(forgetting_factor * np.eye(m) + phi.T @
        # P_phi)```
        S = forgetting_factor * np.eye(m) + phi.T @ P_phi
        K = linalg.solve(S, P_phi.T, assume_a='pos').T

        # 2. Update parameters
        self._parameter_cache = theta + K @ error

        # 3. Update covariance matrix P
        # Since P is symmetric, phi.T @ p is equivalent to P_phi.T
        p = (p - K @ P_phi.T) / forgetting_factor
        # Enforce symmetry to avoid numerical drift
        self._p = (p + p.T) / 2.0

    def update(self):
        if self._parameter_cache is not None:
            self.set_parameter(self._parameter_cache)
            self._parameter_cache = None
