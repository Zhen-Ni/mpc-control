#!/usr/bin/env python3
"""Model Predictive Control (MPC) module.

Provides the `Mpc` class for formulating and solving quadratic
programming (QP) problems using OSQP for discrete-time systems
(including linear, affine, and linearized nonlinear systems).
"""

from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csc_array

import osqp

from .discrete import Discrete, AffineTimeInvariant


def block_diag(arrays, format):
    """Build a block diagonal sparse array from provided arrays.

    This is a protocol for scipy.sparse.block_diag, as the scipy's
    version would be confused and provide warnings if arrays are all
    dense.

    This function can be deprecated when scipy sparse array interfaces
    are stable. (No warnings will be printed then)
    """
    array_list = []
    for i, a in enumerate(arrays):
        if i == 0:
            array_list.append(csc_array(a))
        else:
            array_list.append(a)
    return sparse.block_diag(array_list, format=format)


def _build_csc_delta_matrix(qp_dim: int, n_control: int) -> csc_array:
    """
    Build the matrix to transfer control input to delta input.

    Return a sparse version of D_bar s.t. ΔU = D_bar * U
    """
    d = np.ones(qp_dim)
    diff = -np.ones(qp_dim - n_control)
    csc_d = sparse.diags(diagonals=[d, diff],
                         offsets=[0, -n_control],
                         shape=(qp_dim, qp_dim),
                         format="csc"
                         )
    return csc_d


@dataclass
class _QpInternal:
    c_bar_m_x: np.ndarray         # c_bar @ m_x
    c_bar_m_u: np.ndarray         # c_bar @ m_u
    c_bar_m_offset: np.ndarray    # c_bar @ m_w @ w + V
    c_bar_m_u_t_q_bar: np.ndarray  # c_bar_m_u.T @ q_bar
    csc_p: csc_array
    q: np.ndarray


@dataclass
class _QpConstraint:
    modified: bool
    output_bound: Optional[tuple[np.ndarray,
                                 np.ndarray,
                                 csc_array]]
    control_bound: Optional[tuple[np.ndarray,
                                  np.ndarray,
                                  csc_array]]
    control_delta_bound: Optional[tuple[np.ndarray,
                                        np.ndarray,
                                        csc_array]]
    control_delta_csc_a: Optional[csc_array]
    a: csc_array
    lb: np.ndarray
    ub: np.ndarray

    @staticmethod
    def new(dim: int) -> _QpConstraint:
        return _QpConstraint(False,
                             None, None, None,
                             None,
                             csc_array(np.zeros([0, dim])),
                             np.zeros([0]),
                             np.zeros([0]))


class Mpc:
    """Model Predictive Controller.

    For discrete time-invariant systems:
        x[n+1] = A * x[n] + B * u[n] + w
        y[n]   = C * x[n] + v
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
        system: Discrete,
        horizon: int,
        output_weighting: np.ndarray,
        control_weighting: Optional[np.ndarray] = None,
        control_delta_weighting: Optional[np.ndarray] = None,
    ):
        """
        Initialize the mpc solver.

        Args:
            system: The discrete system object.
            horizon: Prediction horizon (N).
            output_weighting: (horizon, n_output, n_output).
            control_weighting: (horizon, n_control, n_control).
            control_delta_weighting: (horizon, n_control, n_control).
        """
        self._system = system
        self._horizon = horizon
        self._mpc_dim = self._horizon * self._system.n_control

        # Type declarations for internal variables
        self._output_weighting: np.ndarray
        self._control_weighting: Optional[np.ndarray]
        self._control_delta_weighting: Optional[np.ndarray]
        self._csc_output_weighting: csc_array
        self._csc_control_weighting: Optional[csc_array]
        self._csc_d_bar_t_r_delta_bar: Optional[csc_array]
        self._csc_control_delta_p: Optional[csc_array]

        self._csc_d_bar = _build_csc_delta_matrix(self._mpc_dim,
                                                  self._system.n_control)
        self._qp_internal: Optional[_QpInternal] = None
        self._qp_constraint = _QpConstraint.new(self._mpc_dim)

        self._result = None
        self._osqp: Optional[osqp.OSQP] = None

        self.set_output_weighting(output_weighting)
        self.set_control_weighting(control_weighting)
        self.set_control_delta_weighting(control_delta_weighting)

    def set_output_weighting(self, output_weighting: np.ndarray) -> None:
        """Set the output weighting matrix."""
        n_output = self._system.n_output
        if output_weighting.shape != (self._horizon, n_output, n_output):
            raise ValueError(
                'shape of `output_weighting` should be '
                f'{(self._horizon, n_output, n_output)}, got '
                f'{output_weighting.shape}')
        self._output_weighting = output_weighting
        self._csc_output_weighting = block_diag(
            output_weighting, format='csc')
        # Invalidate the cached QP matrices to force rebuild in next solve()
        self._qp_internal = None

    def set_control_weighting(
            self, control_weighting: Optional[np.ndarray] = None) -> None:
        """Set the control weighting matrix."""
        n_control = self._system.n_control
        if (control_weighting is not None) and \
           (control_weighting.shape !=
                (self._horizon, n_control, n_control)):
            raise ValueError(
                'shape of `control_weighting` should be '
                f'{(self._horizon, n_control, n_control)}, got '
                f'{control_weighting.shape}')
        self._control_weighting = control_weighting
        if control_weighting is None:
            self._csc_control_weighting = None
        else:
            self._csc_control_weighting = block_diag(
                control_weighting, format='csc')
        self._qp_internal = None

    def set_control_delta_weighting(
            self,
            control_delta_weighting: Optional[np.ndarray] = None) -> None:
        """Set the control delta weighting matrix."""
        n_control = self._system.n_control
        if (control_delta_weighting is not None) and \
           (control_delta_weighting.shape !=
                (self._horizon, n_control, n_control)):
            raise ValueError(
                'shape of `control_delta_weighting` should be '
                f'{(self._horizon, n_control, n_control)}, got '
                f'{control_delta_weighting.shape}')
        self._control_delta_weighting = control_delta_weighting
        if control_delta_weighting is None:
            self._csc_d_bar_t_r_delta_bar = None
            self._csc_control_delta_p = None
        else:
            csc_control_delta_weighting = block_diag(
                control_delta_weighting, format='csc')
            self._csc_d_bar_t_r_delta_bar = self._csc_d_bar.T @ \
                csc_control_delta_weighting
            self._csc_control_delta_p = self._csc_d_bar_t_r_delta_bar @ \
                self._csc_d_bar
        self._qp_internal = None

    @property
    def result(self):
        """Return the lastest verbose result."""
        return self._result

    def set_output_limit(self,
                         lb: np.ndarray,
                         ub: np.ndarray,
                         proj: Optional[np.ndarray] = None
                         ) -> None:
        """Set the limit of output values.

        Args:
            lb: The lower bound with shape (horizon, m).
            ub: The upper bound with shape (horizon, m).
            proj: The linear transformation matrix for output
                with shape (horizon, m, n_output). Default
                to identity matrix (m = n_output).
        """
        if lb.shape != ub.shape:
            raise ValueError(
                f'shape of lb ({lb.shape}) and ub ({ub.shape}) must match')
        if lb.ndim != 2 or lb.shape[0] != self._horizon:
            raise ValueError(
                f'shape of lb/ub should be ({self._horizon}, m), '
                f'got {lb.shape}')

        m = lb.shape[1]
        n_output = self._system.n_output

        if proj is None:
            if m != n_output:
                raise ValueError(f'proj must be provided if m ({m}) '
                                 f'!= n_output ({n_output})')
            proj = np.stack([np.eye(n_output)] * self._horizon)
        elif proj.shape != (self._horizon, m, n_output):
            raise ValueError('shape of proj should be '
                             f'{(self._horizon, m, n_output)}, '
                             f'got {proj.shape}')

        csc_trans = block_diag(proj, format='csc')
        self._qp_constraint.modified = True
        self._qp_constraint.output_bound = lb, ub, csc_trans

    def set_control_limit(self,
                          lb: np.ndarray,
                          ub: np.ndarray,
                          proj: Optional[np.ndarray] = None
                          ) -> None:
        """Set the input limit of control vectors.

        Args:
            lb: The lower bound with shape (horizon, m).
            ub: The upper bound with shape (horizon, m).
            proj: The linear transformation matrix for control
                with shape (horizon, m, n_control). Default
                to identity matrix (m = n_control).
        """
        if lb.shape != ub.shape:
            raise ValueError(
                f'shape of lb ({lb.shape}) and ub ({ub.shape}) must match')
        if lb.ndim != 2 or lb.shape[0] != self._horizon:
            raise ValueError(
                f'shape of lb/ub should be ({self._horizon}, m), '
                f'got {lb.shape}')

        m = lb.shape[1]
        n_control = self._system.n_control

        if proj is None:
            if m != n_control:
                raise ValueError(f'proj must be provided if m ({m})'
                                 f' != n_control ({n_control})')
            proj = np.stack([np.eye(n_control)] * self._horizon)
        elif proj.shape != (self._horizon, m, n_control):
            raise ValueError('shape of proj should be '
                             f'{(self._horizon, m, n_control)}, '
                             f'got {proj.shape}')

        csc_trans = block_diag(proj, format='csc')
        self._qp_constraint.modified = True
        self._qp_constraint.control_bound = lb, ub, csc_trans

    def set_control_rate_limit(self,
                               lb: np.ndarray,
                               ub: np.ndarray,
                               proj: Optional[np.ndarray] = None
                               ) -> None:
        """Set the limit of control changing rate.

        Args:
            lb: The lower bound with shape (horizon, m).
            ub: The upper bound with shape (horizon, m).
            proj: The linear transformation matrix for control rate
                with shape (horizon, m, n_control). Default
                to identity matrix (m = n_control).
        """
        if lb.shape != ub.shape:
            raise ValueError(
                f'shape of lb ({lb.shape}) and ub ({ub.shape}) must match')
        if lb.ndim != 2 or lb.shape[0] != self._horizon:
            raise ValueError(
                f'shape of lb/ub should be ({self._horizon}, m), '
                f'got {lb.shape}')

        m = lb.shape[1]
        n_control = self._system.n_control

        if proj is None:
            if m != n_control:
                raise ValueError(f'proj must be provided if m ({m})'
                                 f' != n_control ({n_control})')
            proj = np.stack([np.eye(n_control)] * self._horizon)
        elif proj.shape != (self._horizon, m, n_control):
            raise ValueError('shape of proj should be '
                             f'{(self._horizon, m, n_control)}, '
                             f'got {proj.shape}')

        csc_trans = block_diag(proj, format='csc')
        control_delta_csc_a = csc_trans @ self._csc_d_bar
        self._qp_constraint.modified = True
        self._qp_constraint.control_delta_bound = lb, ub, csc_trans
        self._qp_constraint.control_delta_csc_a = control_delta_csc_a

    def _build_constraints(self,
                           initial_state: np.ndarray,
                           previous_control: np.ndarray) -> None:
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
            lb, ub, csc_trans = self._qp_constraint.control_bound
            a = csc_trans
            lb_list.append(lb.reshape(-1))
            ub_list.append(ub.reshape(-1))
            a_list.append(a)

        if self._qp_constraint.output_bound:
            lb, ub, csc_trans = self._qp_constraint.output_bound
            a = csc_trans @ self._qp_internal.c_bar_m_u
            offset = csc_trans @ (self._qp_internal.c_bar_m_x @ initial_state +
                                  self._qp_internal.c_bar_m_offset)
            lb = lb.reshape(-1) - offset
            ub = ub.reshape(-1) - offset
            a_list.append(csc_array(a))
            lb_list.append(lb)
            ub_list.append(ub)

        if self._qp_constraint.control_delta_bound:
            lb, ub, csc_trans = self._qp_constraint.control_delta_bound
            control_offset = csc_trans[:, :self._system.n_control] @ \
                previous_control
            a = self._qp_constraint.control_delta_csc_a
            lb = lb.reshape(-1) + control_offset
            ub = ub.reshape(-1) + control_offset
            a_list.append(a)
            lb_list.append(lb)
            ub_list.append(ub)

        if a_list:
            self._qp_constraint.a = sparse.vstack(a_list, format='csc')
            self._qp_constraint.lb = np.concatenate(lb_list)
            self._qp_constraint.ub = np.concatenate(ub_list)
        else:
            self._qp_constraint.a = csc_array(np.zeros([0, self._mpc_dim]))
            self._qp_constraint.lb = np.zeros([0])
            self._qp_constraint.ub = np.zeros([0])
        self._qp_constraint.modified = False

    def _update_constraints(self,
                            initial_state: np.ndarray,
                            previous_control: np.ndarray) -> None:
        """Update the constraint bounds in self._qp_internal.

        This method assumes `self._qp_constraint.modified` is False
        and the system is AffineTimeInvariant. Only the bounds
        dependent on `initial_state` and `previous_control` are
        recalculated.

        """
        if self._qp_internal is None:
            raise ValueError('self._qp_internal should be built first')

        lb_list = []
        ub_list = []

        if self._qp_constraint.control_bound:
            lb, ub, _ = self._qp_constraint.control_bound
            # Control bounds do not depend on state or previous control
            lb_list.append(lb.reshape(-1))
            ub_list.append(ub.reshape(-1))

        if self._qp_constraint.output_bound:
            lb, ub, csc_trans = self._qp_constraint.output_bound
            offset = csc_trans @ (self._qp_internal.c_bar_m_x @ initial_state +
                                  self._qp_internal.c_bar_m_offset)
            lb_list.append(lb.reshape(-1) - offset)
            ub_list.append(ub.reshape(-1) - offset)

        if self._qp_constraint.control_delta_bound:
            lb, ub, csc_trans = self._qp_constraint.control_delta_bound
            control_offset = csc_trans[:, :self._system.n_control] @ \
                previous_control
            lb = lb.reshape(-1) + control_offset
            ub = ub.reshape(-1) + control_offset
            lb_list.append(lb)
            ub_list.append(ub)

        if lb_list:
            self._qp_constraint.lb = np.concatenate(lb_list)
            self._qp_constraint.ub = np.concatenate(ub_list)
        else:
            self._qp_constraint.lb = np.zeros([0])
            self._qp_constraint.ub = np.zeros([0])

    def _assemble_linear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            c_bar_m_x, c_bar_m_u, c_bar_m_offset):
        """Assemble the qp intermediate matrixes inplace."""
        a = self._system.transition_matrix
        b = self._system.control_matrix
        w = self._system.state_disturbance_vector
        c = self._system.output_matrix
        v = self._system.output_disturbance_vector

        c_a_powers = np.zeros([self._horizon + 1, n_output, n_state])
        c_a_powers[0] = c
        c_a_sum = np.zeros([n_output, n_state])

        for i in range(self._horizon):
            c_a_powers[i+1] = c_a_powers[i] @ a

            # Build c_bar_m_x, c_bar_m_u, c_bar_m_d_d
            start_idx = n_output * i
            stop_idx = start_idx + n_output

            # c_bar_m_x
            c_bar_m_x[start_idx: stop_idx, :] = c_a_powers[i+1]

            # c_bar_m_u
            c_bar_m_u[start_idx: stop_idx, : (i+1)*n_control] = \
                np.einsum('ijk,kl->jil',
                          c_a_powers[i::-1],
                          b).reshape(n_output, -1)

            # c_bar_m_offset
            c_a_sum += c_a_powers[i]
            c_bar_m_offset[start_idx: stop_idx] = (c_a_sum @ w + v).reshape(-1)

    def _assemble_nonlinear_qp_helper(
            self,
            state_ref, control_ref, n_state, n_control, n_output,
            c_bar_m_x, c_bar_m_u, c_bar_m_offset):
        """Assemble the qp intermediate matrixes inplace."""
        if state_ref is None:
            state_ref = [None] * self._horizon
        if control_ref is None:
            control_ref = [None] * self._horizon

        m_w = np.zeros([self._horizon, self._horizon,
                        n_state, n_state])
        c_bar_m_w_i = np.zeros([self._horizon, n_output, n_state])
        a_0 = np.zeros([n_state, n_state])

        # Store control_matrix and state disturbance vectors.
        bs = np.zeros([self._horizon, n_state, n_control])
        ws = np.zeros([self._horizon, n_state])

        for i in range(self._horizon):
            system = self._system.linearize(state_ref[i], control_ref[i])
            a_i = system.transition_matrix
            b_i = system.control_matrix
            w_i = system.state_disturbance_vector
            c_i = system.output_matrix
            v_i = system.output_disturbance_vector

            # Build m_w
            m_w[i, i] = np.eye(n_state)
            # Use vectorized operation for better performance
            # for j in range(i):
            #     m_w[i, j] = a_i @ m_w[i - 1, j]
            m_w[i, :i] = a_i @ m_w[i - 1, :i]

            # Cache c_bar_m_w_i for this horizon.
            # Use vectorized operation for better performance
            # for j in range(i + 1):
            #     c_bar_m_w_i[j] = c_i @ m_w[i, j]
            c_bar_m_w_i[:i+1] = c_i @ m_w[i, :i + 1]

            # Build c_bar_m_x
            if i == 0:
                a_0 = a_i
            start_idx = n_output * i
            stop_idx = start_idx + n_output
            c_bar_m_x[start_idx: stop_idx, :] = c_bar_m_w_i[0] @ a_0

            # Build c_bar_m_u
            bs[i] = b_i
            # Use vectorized operation for better performance.
            # for j in range(i + 1):
            #     row_start_idx = n_control * j
            #     row_stop_idx = row_start_idx + n_control
            #     c_bar_m_u[start_idx: stop_idx, \
            #         row_start_idx: row_stop_idx] = \
            #         c_bar_m_w_i[j] @ bs[j]
            c_bar_m_u[start_idx: stop_idx, :n_control*(i+1)] = \
                np.einsum('ijk,ikl->jil',
                          c_bar_m_w_i[:i+1],
                          bs[:i+1]).reshape(n_output, -1)

            # Build c_bar_m_offset
            ws[i] = w_i
            # Use vectorized operation for better performance.
            # c_bar_m_offset[start_idx: stop_idx] = \
            # sum([c_bar_m_w_i[j] @ ws[j] for j in range(i+1)]) + v_i
            c_bar_m_offset[start_idx: stop_idx] = \
                np.einsum('ijk,ik->j', c_bar_m_w_i[:i+1], ws[:i+1]) + v_i

    def _build_qp(self,
                  target_output: np.ndarray,
                  initial_state: np.ndarray,
                  previous_control: np.ndarray,
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

           X = M_x * x_0 + M_u * U + M_w * w
           Y = C_bar * X + V
             = C_bar * M_x * x_0 + C_bar * M_u * U + C_bar * M_w * w + V

        2. Cost function:
           J = (Y - Y_ref)^T * Q_bar * (Y - Y_ref)
             + U^T * R_bar * U
             + ΔU^T * R_Δ_bar * ΔU

           where the control delta is ΔU = D_bar * U - U_last
           U_last = [u_{-1}^T, 0, ..., 0]^T (u_{-1} is the
           previous control input)

        3. OSQP standard form (min 0.5 * U^T * P * U + q^T * U):
          Let E_y = C_bar * M_x * x_0 + C_bar * M_w * w + V - Y_ref
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
            previous_control: The control input in the previous
                timestep u_{-1}.
            state_ref: Reference state for system linearization.
            control_ref: Reference control for system linearization.

        """
        n_state = self._system.n_state
        n_control = self._system.n_control
        n_output = self._system.n_output

        n_total_output = self._horizon * n_output

        # Build M_x, M_u, C_bar
        c_bar_m_x = np.zeros([n_total_output, n_state])
        c_bar_m_u = np.zeros([n_total_output, self._mpc_dim])
        c_bar_m_offset = np.zeros([n_total_output, ])
        if isinstance(self._system, AffineTimeInvariant):
            self._assemble_linear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                c_bar_m_x, c_bar_m_u, c_bar_m_offset)
        else:
            self._assemble_nonlinear_qp_helper(
                state_ref, control_ref, n_state, n_control, n_output,
                c_bar_m_x, c_bar_m_u, c_bar_m_offset)

        # Build internal matrixes
        csc_q_bar = self._csc_output_weighting

        # Calculate P and q for output (y).
        y_ref_vec = target_output.reshape(-1)
        e_y = (c_bar_m_x @ initial_state +
               c_bar_m_offset).reshape(-1) - y_ref_vec
        c_bar_m_u_t_q_bar = c_bar_m_u.T @ csc_q_bar
        p = c_bar_m_u_t_q_bar @ c_bar_m_u
        q = c_bar_m_u_t_q_bar @ e_y

        if self._control_weighting is not None:
            csc_r_bar = self._csc_control_weighting
            p += csc_r_bar
        else:
            csc_r_bar = None

        if self._control_delta_weighting is not None:
            csc_d_bar_t_r_delta_bar = self._csc_d_bar_t_r_delta_bar
            # To satisfy mypy
            assert csc_d_bar_t_r_delta_bar is not None
            p += self._csc_control_delta_p
            q -= csc_d_bar_t_r_delta_bar[:, :n_control] @ previous_control

        self._qp_internal = _QpInternal(
            c_bar_m_x, c_bar_m_u, c_bar_m_offset, c_bar_m_u_t_q_bar,
            csc_array(p), q, )

    def _update_qp(self,
                   target_output: np.ndarray,
                   initial_state: np.ndarray,
                   previous_control: np.ndarray,
                   ) -> None:
        """Update the qp problem with minimal effort.

        This method should only be used when initial state or target
        output is changed. Do not rely on this method if the system
        matrixes or disturbance vectors are changed.

        """
        if self._qp_internal is None:
            raise ValueError(
                '`_update_qp` can only be called after the '
                'problem is built by `_build_qp`')

        c_bar_m_x = self._qp_internal.c_bar_m_x
        c_bar_m_offset = self._qp_internal.c_bar_m_offset
        c_bar_m_u_t_q_bar = self._qp_internal.c_bar_m_u_t_q_bar
        y_ref_vec = target_output.reshape(-1)

        e_y = (c_bar_m_x @ initial_state +
               c_bar_m_offset).reshape(-1) - y_ref_vec
        q = c_bar_m_u_t_q_bar @ e_y

        if self._control_delta_weighting is not None:
            n_control = self._system.n_control
            csc_d_bar_t_r_delta_bar = self._csc_d_bar_t_r_delta_bar
            # To satisfy mypy
            assert csc_d_bar_t_r_delta_bar is not None
            q -= csc_d_bar_t_r_delta_bar[:, :n_control] @ \
                previous_control

        self._qp_internal.q = q

    def _validate_previous_control(
            self,
            previous_control: Optional[np.ndarray]) -> np.ndarray:
        if previous_control is not None:
            return previous_control
        # Use a placeholder for previous control, as it will not be
        # used in this case.
        if (self._control_delta_weighting is None and
                self._qp_constraint.control_delta_bound is None):
            return np.empty(self._system.n_control)
        raise ValueError(
            'previous_control must be provided if '
            'control changing rate is involved.')

    def _prepare_qp_matrices(
            self,
            target_output: np.ndarray,
            initial_state: np.ndarray,
            previous_control: np.ndarray,
            state_ref: Optional[np.ndarray],
            control_ref: Optional[np.ndarray],
            use_cached_p: bool) -> None:
        if use_cached_p:
            self._update_qp(target_output, initial_state, previous_control)
        else:
            self._build_qp(target_output, initial_state, previous_control,
                           state_ref, control_ref)

    def _prepare_constraints(
            self,
            initial_state: np.ndarray,
            previous_control: np.ndarray,
            use_cached_a: bool) -> None:
        if use_cached_a:
            self._update_constraints(initial_state, previous_control)
        else:
            self._build_constraints(initial_state, previous_control)

    def _setup_and_solve_osqp(
            self,
            max_iter: Optional[int],
            eps_abs: Optional[float],
            eps_rel: Optional[float],
            control_ref: Optional[np.ndarray],
            use_cached_p: bool,
            use_cached_a: bool) -> Optional[np.ndarray]:
        assert self._qp_internal is not None  # Make mypy happy

        p = self._qp_internal.csc_p
        q = self._qp_internal.q
        a = self._qp_constraint.a
        lb = self._qp_constraint.lb
        ub = self._qp_constraint.ub

        if self._osqp is None or not use_cached_p or not use_cached_a:
            if self._osqp is None:
                self._osqp = osqp.OSQP()
            kwargs = dict(max_iter=max_iter,
                          eps_abs=eps_abs,
                          eps_rel=eps_rel)
            kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
            self._osqp.setup(p, q, a, lb, ub,
                             warm_starting=True, verbose=False,
                             **kwargs)
        else:
            self._osqp.update(q=q, l=lb, u=ub)
            if max_iter is not None:
                self._osqp.update_settings(max_iter=max_iter)
            if eps_abs is not None:
                self._osqp.update_settings(eps_abs=eps_abs)
            if eps_rel is not None:
                self._osqp.update_settings(eps_rel=eps_rel)

        if control_ref is not None:
            self._osqp.warm_start(x=np.asarray(control_ref).reshape(-1))

        res = self._osqp.solve(raise_error=False)
        self._result = res
        if res.info.status == 'solved':
            return res.x.reshape(self._horizon, self._system.n_control)
        return None

    def solve(self,
              target_output: np.ndarray,
              initial_state: np.ndarray,
              previous_control: Optional[np.ndarray] = None,
              state_ref: Optional[np.ndarray] = None,
              control_ref: Optional[np.ndarray] = None,
              max_iter: Optional[int] = None,
              eps_abs: Optional[float] = None,
              eps_rel: Optional[float] = None
              ) -> Optional[np.ndarray]:
        """Solve the mpc problem.

        Args:
            initial_state: Initial state (x_0).
            previous_control: The control input in the previous
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
        use_cached_p = bool(self._qp_internal and
                            isinstance(self._system, AffineTimeInvariant))
        use_cached_a = bool((not self._qp_constraint.modified) and
                            isinstance(self._system, AffineTimeInvariant))

        previous_control = self._validate_previous_control(previous_control)

        self._prepare_qp_matrices(
            target_output, initial_state, previous_control,
            state_ref, control_ref, use_cached_p)

        self._prepare_constraints(
            initial_state, previous_control, use_cached_a)

        return self._setup_and_solve_osqp(
            max_iter, eps_abs, eps_rel, control_ref,
            use_cached_p, use_cached_a)
