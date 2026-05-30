#!/usr/bin/env python3

from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix

import osqp

from discrete import DiscreteSystem, LtiSystem


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
        initial_state: np.ndarray,
        target_output: np.ndarray,
        output_weighting_matrix: np.ndarray,
        control_weighting_matrix: np.ndarray
    ):
        """
        Initialize the mpc solver.

        Args:
            system: The discrete LTI system object.
            horizon: Prediction horizon (N).
            initial_state: Initial state.
            target_output: Reference output sequence.
            output_weighting_matrix: (n_output, n_output)
            control_weighting_matrix: (n_control, n_control).
        """
        self._system = system
        self._horizon = horizon
        self._initial_state = initial_state
        self._target_output = target_output
        self._output_weighting_matrix = output_weighting_matrix
        self._control_weighting_matrix = control_weighting_matrix
        self._mpc_dim = self._horizon * self._system.n_control
        
    def _assemble_linear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            m_x, m_u, c_bar, q_bar, r_bar):
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

            # Build Q_bar
            q_bar[start_idx_y: stop_idx_y,
                  start_idx_y: stop_idx_y
                  ] = self._output_weighting_matrix

            # Build R_bar
            r_bar[start_idx_u: stop_idx_u,
                  start_idx_u: stop_idx_u
                  ] = self._control_weighting_matrix

            # Update a_last for next iteration
            a_last = m_x[start_idx_x: stop_idx_x]

    def _assemble_nonlinear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            m_x, m_u, c_bar, q_bar, r_bar):
        """Assemble the qp intermediate matrixes inplace."""
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

            # Build Q_bar
            q_bar[start_idx_y: stop_idx_y,
                  start_idx_y: stop_idx_y
                  ] = self._output_weighting_matrix

            # Build R_bar
            r_bar[start_idx_u: stop_idx_u,
                  start_idx_u: stop_idx_u
                  ] = self._control_weighting_matrix

            # Update a_last for next iteration
            a_last = m_x[start_idx_x: stop_idx_x]

    def _build_qp(self,
                  state_ref: list[Optional[np.ndarray]],
                  control_ref: list[Optional[np.ndarray]]):
        """
        Build the standart quadratic programming problem.

        Build the P and q matrices of the QP problem
        based on current state, output reference
        trajectory, and weight matrices.

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

        3. Standard QP form
           (min 0.5 * U^T * P * U + q^T * U):
           Let E_y = C_bar * M_x * x_0 - Y_ref
           P = 2 * (M_u^T * C_bar^T * Q_bar * C_bar
                  * M_u + R_bar)
           q = 2 * M_u^T * C_bar^T * Q_bar * E_y

        Args:
            state_ref: reference state for system linearization.
            control_ref: reference state for system linearization.
        """

        # 1. Build M_x, M_u, and C_bar
        n_state = self._system.n_state
        n_control = self._system.n_control
        n_output = self._system.n_output

        n_total_state = self._horizon * n_state
        n_total_output = self._horizon * n_output
        n_total_control = self._mpc_dim

        m_x = np.zeros([n_total_state, n_state])
        m_u = np.zeros([n_total_state, self._mpc_dim])
        c_bar = np.zeros([n_total_output, n_total_state])
        q_bar = np.zeros([n_total_output, n_total_output])
        r_bar = np.zeros([n_total_control, n_total_control])

        if isinstance(self._system, LtiSystem):
            self._assemble_linear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                m_x, m_u, c_bar, q_bar, r_bar)
        else:
            self._assemble_nonlinear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                m_x, m_u, c_bar, q_bar, r_bar)

        # 3. Calculate P and q based on output error
        c_bar_m_x = c_bar @ m_x
        c_bar_m_u = c_bar @ m_u
        y_ref_vec = self._target_output.reshape(-1)
        e_y = (c_bar_m_x @ self._initial_state).reshape(-1) - y_ref_vec
        p = 2 * (c_bar_m_u.T @ q_bar @ c_bar_m_u + r_bar)
        q = 2 * c_bar_m_u.T @ q_bar @ e_y

        return p, q

    def solve(self,
              state_ref: Optional[
                  list[Optional[np.ndarray]] |
                  np.ndarray] = None,
              control_ref: Optional[
                  list[Optional[np.ndarray]] |
                  np.ndarray] = None,
              max_iter: Optional[int] = None,
              eps_abs: Optional[float] = None,
              eps_rel: Optional[float] = None
              ):
        """
        Solve the mpc problem.

        Args:
            state_ref: reference state for system linearization.
            control_ref: reference state for system linearization.
        """
        warm_start = True
        if state_ref is None:
            state_ref = [None] * self._horizon
        if control_ref is None:
            control_ref = [None] * self._horizon
            warm_start = False

        p, q = self._build_qp(list(state_ref), list(control_ref))
        p_sparse = csc_matrix(p)

        prob = osqp.OSQP()
        a = csc_matrix((0, q.shape[0]))
        lb = []
        ub = []
        kwargs = dict(max_iter=max_iter,
                      eps_abs=eps_abs,
                      eps_rel=eps_rel)
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
        prob.setup(p_sparse, q, a, lb, ub,
                   warm_start=warm_start, verbose=False, **kwargs)
        if warm_start:
            prob.warm_start(np.asarray(control_ref).reshape(-1))
        res = prob.solve()
        u = res.x.reshape(self._horizon, self._system.n_control)
        return u
