#!/usr/bin/env python3

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csc_matrix

import osqp

from .discrete import DiscreteSystem, LtiSystem


def _build_delta_matrix(qp_dim, n_control) -> np.ndarray:
    """
    Build the matrix to transfer control input to delta input.

    Return D_bar s.t. ΔU = D_bar * U
    """
    d = np.eye(qp_dim)
    dif = np.diag(np.ones(qp_dim-n_control), k=-n_control)
    return d - dif


@dataclass
class _QpInternal:
    m_x: np.ndarray
    m_u: np.ndarray
    c_bar: np.ndarray
    q_bar: np.ndarray
    r_bar: Optional[np.ndarray]
    r_delta_bar: Optional[np.ndarray]
    c_bar_m_x: np.ndarray
    c_bar_m_u: np.ndarray
    p: np.ndarray
    q: np.ndarray


@dataclass
class _QpConstraint:
    modified: bool
    output_bound: Optional[tuple[np.ndarray, np.ndarray]]
    control_bound: Optional[tuple[np.ndarray, np.ndarray]]
    control_delta_bound: Optional[tuple[np.ndarray, np.ndarray]]
    a: np.ndarray
    lb: np.ndarray
    ub: np.ndarray

    @staticmethod
    def new(dim: int) -> _QpConstraint:
        return _QpConstraint(False,
                             None, None, None,
                             np.zeros([0, dim]),
                             np.zeros([0]),
                             np.zeros([0]))


class Mpc:
    """Model Predictive Controller.

    For discrete time-invariant systems:
        x[n+1] = A * x[n] + B * u[n]
        y[n]   = C * x[n]
    Formulates and solves a QP problem using OSQP:
        min  0.5 * U^T * P * U + q^T * U
        s.t. l <= A_c * U <= u
    where U is the sequence of control inputs over the prediction
    horizon.

    Q is the output weighting matrix penalizing deviation from the
    output reference. R is the control weighting matrix penalizing
    control effort.
    """

    def __init__(
        self,
        system: DiscreteSystem,
        horizon: int,
        output_weighting: np.ndarray,
        control_weighting: Optional[np.ndarray] = None,
        control_delta_weighting: Optional[np.ndarray] = None,
    ):
        """
        Initialize the mpc solver.

        Args:
            system: The discrete LTI system object.
            horizon: Prediction horizon (N).
            output_weighting: (n_output, n_output).
            control_weighting: (n_control, n_control).
            control_delta_weighting: (n_control, n_control).
        """
        self._system = system
        self._horizon = horizon
        self._output_weighting = output_weighting
        self._control_weighting = control_weighting
        self._control_delta_weighting = control_delta_weighting
        self._mpc_dim = self._horizon * self._system.n_control

        self._qp_internal: Optional[_QpInternal] = None
        self._qp_constraint = _QpConstraint.new(self._mpc_dim)

        self._result = None

    @property
    def result(self):
        """Return the lastest verbose result."""
        return self._result

    def set_output_limit(self,
                         lb: np.ndarray,
                         ub: np.ndarray) -> None:
        """Set the limit of output values.

        Args:
            lb: The lower bound with shape (horizon, n_output).
            ub: The upper bound with shape (horizon, n_output).
        """
        dim = (self._horizon, self._system.n_output)
        if not lb.shape == dim:
            raise ValueError(f'shape of lb should be {dim}, '
                             f'got {lb.shape}')
        if not ub.shape == dim:
            raise ValueError(f'shape of ub should be {dim}, '
                             f'got {ub.shape}')
        self._qp_constraint.modified = True
        self._qp_constraint.output_bound = lb, ub

    def set_control_limit(self,
                          lb: np.ndarray,
                          ub: np.ndarray) -> None:
        """Set the input limit of control vectors.

        Args:
            lb: The lower bound with shape (horizon, n_control).
            ub: The upper bound with shape (horizon, n_control).
        """
        dim = (self._horizon, self._system.n_control)
        if not lb.shape == dim:
            raise ValueError(f'shape of lb should be {dim}, '
                             f'got {lb.shape}')
        if not ub.shape == dim:
            raise ValueError(f'shape of ub should be {dim}, '
                             f'got {ub.shape}')
        self._qp_constraint.modified = True
        self._qp_constraint.control_bound = lb, ub

    def set_control_rate_limit(self,
                               lb: np.ndarray,
                               ub: np.ndarray) -> None:
        """Set the limit of control changing rate.

        Args:
            lb: The lower bound with shape (horizon, n_control).
            ub: The upper bound with shape (horizon, n_control).
        """
        dim = (self._horizon, self._system.n_control)
        if not lb.shape == dim:
            raise ValueError(f'shape of lb should be {dim}, '
                             f'got {lb.shape}')
        if not ub.shape == dim:
            raise ValueError(f'shape of ub should be {dim}, '
                             f'got {ub.shape}')
        self._qp_constraint.modified = True
        self._qp_constraint.control_delta_bound = lb, ub

    def _build_constraints(self,
                           initial_state: np.ndarray,
                           initial_control: np.ndarray) -> None:
        """Build the constraints in self._qp_internal.

        Make sure self._qp_internal is correctly initialized before
        calling this.

        """
        if self._qp_internal is None:
            raise ValueError('self._qp_internal should be built first')
        a_list = []
        lb_list = []
        ub_list = []
        if self._qp_constraint.control_bound:
            a_list.append(np.eye(self._mpc_dim))
            lb_list.append(self._qp_constraint.control_bound[0].reshape(-1))
            ub_list.append(self._qp_constraint.control_bound[1].reshape(-1))

        if self._qp_constraint.output_bound:
            a = self._qp_internal.c_bar_m_u
            offset = self._qp_internal.c_bar_m_x @ initial_state
            lb = self._qp_constraint.output_bound[0].reshape(-1) - offset
            ub = self._qp_constraint.output_bound[1].reshape(-1) - offset
            a_list.append(a)
            lb_list.append(lb)
            ub_list.append(ub)

        if self._qp_constraint.control_delta_bound:
            d_bar = _build_delta_matrix(self._mpc_dim, self._system.n_control)
            control_bar = np.zeros(self._mpc_dim)
            control_bar[:self._system.n_control] = initial_control
            a = d_bar
            lb = (self._qp_constraint.control_delta_bound[0].reshape(-1) +
                  control_bar)
            ub = (self._qp_constraint.control_delta_bound[1].reshape(-1) +
                  control_bar)
            a_list.append(a)
            lb_list.append(lb)
            ub_list.append(ub)

        if a_list:
            self._qp_constraint.a = np.concat(a_list)
            self._qp_constraint.lb = np.concat(lb_list)
            self._qp_constraint.ub = np.concat(ub_list)
        else:
            self._qp_constraint.a = np.zeros([0, self._mpc_dim])
            self._qp_constraint.lb = np.zeros([0])
            self._qp_constraint.ub = np.zeros([0])
        self._qp_constraint.modified = False

    def _assemble_linear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            m_x, m_u, c_bar):
        """Assemble the qp intermediate matrixes inplace."""
        a = self._system.get_transition_matrix()
        b = self._system.get_control_matrix()
        c = self._system.get_output_matrix()

        a_last = np.eye(n_state)
        for i in range(self._horizon):
            start_idx_x = n_state * i
            stop_idx_x = start_idx_x + n_state
            m_x[start_idx_x: stop_idx_x] = a @ a_last

            start_idx_u = i * n_control
            stop_idx_u = start_idx_u + n_control
            m_u[start_idx_x: stop_idx_x, :n_control] = a_last @ b
            if i:
                mu_prev = m_u[start_idx_x - n_state: start_idx_x,
                              :start_idx_u]
                m_u[start_idx_x: stop_idx_x, n_control: stop_idx_u] = mu_prev

            # Build C_bar
            start_idx_y = n_output * i
            stop_idx_y = start_idx_y + n_output
            c_bar[start_idx_y: stop_idx_y, start_idx_x: stop_idx_x] = c

            # Update a_last for next iteration
            a_last = m_x[start_idx_x: stop_idx_x]

    def _assemble_nonlinear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            m_x, m_u, c_bar):
        """Assemble the qp intermediate matrixes inplace."""
        if state_ref is None:
            state_ref = [None] * self._horizon
        if control_ref is None:
            control_ref = [None] * self._horizon

        a_last = np.eye(n_state)
        for i in range(self._horizon):
            system = self._system.linearize(state_ref[i], control_ref[i])
            a_i = system.get_transition_matrix()
            b_i = system.get_control_matrix()
            c_i = system.get_output_matrix()

            start_idx_x = n_state * i
            stop_idx_x = start_idx_x + n_state
            m_x[start_idx_x: stop_idx_x] = a_i @ a_last

            start_idx_u = i * n_control
            stop_idx_u = start_idx_u + n_control
            m_u[start_idx_x: stop_idx_x, start_idx_u: stop_idx_u] = b_i
            if i:
                mu_prev = m_u[start_idx_x - n_state: start_idx_x,
                              :start_idx_u]
                m_u[start_idx_x: stop_idx_x, :start_idx_u] = a_i @ mu_prev

            # Build C_bar
            start_idx_y = n_output * i
            stop_idx_y = start_idx_y + n_output
            c_bar[start_idx_y: stop_idx_y, start_idx_x: stop_idx_x] = c_i

            # Update a_last for next iteration
            a_last = m_x[start_idx_x: stop_idx_x]

    def _build_qp(self,
                  target_output: np.ndarray,
                  initial_state: np.ndarray,
                  initial_control: np.ndarray,
                  state_ref: Optional[np.ndarray],
                  control_ref: Optional[np.ndarray]) -> None:
        """Build the standard quadratic programming problem.

        Build the P and q matrices of the QP problem required by OSQP
        based on current state, output reference trajectory, and
        weight matrices.

        Mathematical Derivation:
        1. System prediction over horizon N:
           X = [x_1^T, x_2^T, ..., x_N^T]^T
           U = [u_0^T, u_1^T, ..., u_{N-1}^T]^T
           Y = [y_1^T, y_2^T, ..., y_N^T]^T

           X = M_x * x_0 + M_u * U
           Y = C_bar * X
             = C_bar * M_x * x_0 + C_bar * M_u * U

        2. Cost function:
           J = (Y - Y_ref)^T * Q_bar * (Y - Y_ref)
             + U^T * R_bar * U
             + ΔU^T * R_Δ_bar * ΔU

           where the control delta is ΔU = D_bar * U - U_last
           U_last = [u_{-1}^T, 0, ..., 0]^T (u_{-1} is the
           previous control input)

        3. OSQP standard form (min 0.5 * U^T * P * U + q^T * U):
           Let E_y = C_bar * M_x * x_0 - Y_ref
           Expanding the cost function and extracting the
        quadratic and linear terms (ignoring constant terms):
           J = 0.5 * U^T * (2 * M_u^T * C_bar^T * Q_bar * C_bar * M_u
                            + 2 * R_bar
                            + 2 * D_bar^T * R_Δ_bar * D_bar) * U
             + (2 * M_u^T * C_bar^T * Q_bar * E_y
                - 2 * D_bar^T * R_Δ_bar * U_last)^T * U

           Matching the OSQP standard form, the actual P and q
           computed in the code are (i.e., without the factor of
           2):
           P = (M_u^T * C_bar^T * Q_bar * C_bar * M_u
                + R_bar
                + D_bar^T * R_Δ_bar * D_bar)
           q = (M_u^T * C_bar^T * Q_bar * E_y
                - D_bar^T * R_Δ_bar * U_last)

        Args:
            target_output: Reference output sequence Y_ref.
            initial_state: Initial state x_0.
            initial_control: The control input in the previous
                timestep u_{-1}.
            state_ref: Reference state for system linearization.
            control_ref: Reference control for system linearization.

        """

        n_state = self._system.n_state
        n_control = self._system.n_control
        n_output = self._system.n_output

        n_total_state = self._horizon * n_state
        n_total_output = self._horizon * n_output
        n_total_control = self._mpc_dim

        # Build M_x, M_u, C_bar
        m_x = np.zeros([n_total_state, n_state])
        m_u = np.zeros([n_total_state, self._mpc_dim])
        c_bar = np.zeros([n_total_output, n_total_state])
        if isinstance(self._system, LtiSystem):
            self._assemble_linear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                m_x, m_u, c_bar)
        else:
            self._assemble_nonlinear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                m_x, m_u, c_bar)

        # Build internal matrixes
        q_bar = np.kron(np.eye(self._horizon),
                        self._output_weighting)

        # Calculate P and q for output (y).
        c_bar_m_x = c_bar @ m_x
        c_bar_m_u = c_bar @ m_u
        y_ref_vec = target_output.reshape(-1)
        e_y = (c_bar_m_x @ initial_state).reshape(-1) - y_ref_vec
        p = c_bar_m_u.T @ q_bar @ c_bar_m_u
        control_bar = np.zeros(self._mpc_dim)
        control_bar[:self._system.n_control] = initial_control
        q = c_bar_m_u.T @ q_bar @ e_y

        if self._control_weighting:
            r_bar = np.kron(np.eye(self._horizon),
                            self._control_weighting)
            p += r_bar
        else:
            r_bar = None

        if self._control_delta_weighting:
            r_delta_bar = np.kron(np.eye(self._horizon),
                                  self._control_delta_weighting)
            d_bar = _build_delta_matrix(n_total_control, n_control)
            control_bar = np.zeros(self._mpc_dim)
            control_bar[:self._system.n_control] = initial_control
            p += d_bar.T @ r_delta_bar @ d_bar
            q -= d_bar.T @ r_delta_bar @ control_bar
        else:
            r_delta_bar = None

        self._qp_internal = _QpInternal(m_x, m_u,
                                        c_bar, q_bar, r_bar,
                                        r_delta_bar,
                                        c_bar_m_x, c_bar_m_u,
                                        p, q)

    def _update_qp(self,
                   target_output: np.ndarray,
                   initial_state: np.ndarray,
                   initial_control: np.ndarray,
                   ) -> None:
        """Update the qp problem with minimal effort.

        This method should only be used when initial state or target
        output is changed. Do not rely on this method if the system
        matrixes are changed.

        """
        if self._qp_internal is None:
            raise ValueError(
                '`_update_qp` can only be called after the '
                'problem is built by `_build_qp`')
        c_bar_m_x = self._qp_internal.c_bar_m_x
        c_bar_m_u = self._qp_internal.c_bar_m_u
        q_bar = self._qp_internal.q_bar
        y_ref_vec = target_output.reshape(-1)
        e_y = (c_bar_m_x @ initial_state).reshape(-1) - y_ref_vec
        q = c_bar_m_u.T @ q_bar @ e_y

        if self._control_delta_weighting:
            d_bar = _build_delta_matrix(self._mpc_dim,
                                        self._system.n_control)
            r_delta_bar = self._qp_internal.r_delta_bar
            # To satisfy mypy
            assert r_delta_bar is not None
            control_bar = np.zeros(self._mpc_dim)
            control_bar[:self._system.n_control] = initial_control
            q -= d_bar.T @ r_delta_bar @ control_bar

        self._qp_internal.q = q

    def solve(self,
              target_output: np.ndarray,
              initial_state: np.ndarray,
              initial_control: Optional[np.ndarray] = None,
              state_ref: Optional[np.ndarray] = None,
              control_ref: Optional[np.ndarray] = None,
              max_iter: Optional[int] = None,
              eps_abs: Optional[float] = None,
              eps_rel: Optional[float] = None
              ) -> Optional[np.ndarray]:
        """Solve the mpc problem.

        Args:
            initial_state: Initial state (x_0).
            initial_control: The control input in the previous
            timestep (u_{-1}).
            target_output: Reference output sequence Y_ref. Note that
                this corresponds to the outputs from step 1 to step N
                (y_1 to y_N), as the output at the current step 0
                (y_0) cannot be influenced by future controls.
            state_ref: Reference state for system linearization. This
                corresponds to the states from step 0 to step N-1 (x_0
                to x_{N-1}).
            control_ref: Reference control for system
                linearization. This corresponds to the controls from
                step 0 to step N-1 (u_0 to u_{N-1}).
            max_iter: Maximum iterations for OSQP solver.
            eps_abs: Absolute convergence tolerance for OSQP solver.
            eps_rel: Relative convergence tolerance for OSQP solver.

        Returns:
            The optimal control sequence U with shape (horizon,
            n_control), corresponding to the controls from step 0 to
            step N-1 (u_0 to u_{N-1}), or None if the problem is not
            solved successfully.

        """
        warm_start = False if control_ref is None else True
        use_cached = self._qp_internal and isinstance(self._system, LtiSystem)

        if initial_control is None:
            if (self._control_delta_weighting is None and
                    self._qp_constraint.control_delta_bound is None):
                initial_control = np.empty(self._system.n_control)
            else:
                raise ValueError(
                    'initial_control must be provided if '
                    'control changing rate is involved.')

        if use_cached:
            self._update_qp(target_output,
                            initial_state, initial_control)
        else:
            self._build_qp(target_output,
                           initial_state, initial_control,
                           state_ref, control_ref)

        self._build_constraints(initial_state, initial_control)

        assert self._qp_internal is not None  # Make mypy happy
        p = self._qp_internal.p
        q = self._qp_internal.q

        p_sparse = csc_matrix(p)

        prob = osqp.OSQP()
        a = csc_matrix(self._qp_constraint.a)
        lb = self._qp_constraint.lb
        ub = self._qp_constraint.ub
        kwargs = dict(max_iter=max_iter,
                      eps_abs=eps_abs,
                      eps_rel=eps_rel)
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
        prob.setup(p_sparse, q, a, lb, ub,
                   warm_starting=warm_start, verbose=False,
                   **kwargs)
        if warm_start:
            res = prob.warm_start(np.asarray(control_ref).reshape(-1))
        res = prob.solve(raise_error=False)
        self._result = res
        if res.info.status == 'solved':
            u = res.x.reshape(self._horizon, self._system.n_control)
            return u
        return None
