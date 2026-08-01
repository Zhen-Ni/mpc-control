#!/usr/bin/env python3

import numpy as np
import mpc_control as mpc

import matplotlib.pyplot as plt
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


def get_discrete_matrices(dt, v, invl, alpha):
    """Vehicle lateral kinematic model with first-order steering lag.

    Continuous state model:
        dy/dt = v * theta
        dtheta/dt = v * invl * phi
        dphi/dt = -alpha * phi + alpha * u
    States: x = [y, theta, phi].
        y: lateral deviation [m]
        theta: heading error [rad]
        phi: actual front wheel steering angle [rad]
    Input: u, steering angle command [rad].
    Discretized using forward Euler approximation.

    Args:
        dt: Discrete sampling time, unit [s].
        v: Longitudinal vehicle speed, unit [m/s].
        invl: Reciprocal of wheelbase (1/L), unit [1/m].
        alpha: Reciprocal of steering lag time constant, unit [1/s].

    Returns:
        tuple:
            A (np.ndarray): State transition matrix, shape (3, 3).
            B (np.ndarray): Input matrix, shape (3, 1).
            C (np.ndarray): Output matrix, shape (2, 3). Observes y and theta.
    """
    A = np.array([[1, dt * v, 0],
                  [0, 1, dt * v * invl],
                  [0, 0, 1 - dt * alpha]])

    B = np.array([[0],
                  [0],
                  [dt * alpha]])
    C = np.array([[1, 0, 0],
                  [0, 1, 0]])
    return A, B, C


class RlsEstimator(mpc.Rls):
    def __init__(self,
                 system: mpc.LtiSystem,
                 delta: float,
                 dt: float,
                 v: float,
                 invl: float,
                 alpha: float
                 ):
        super().__init__(system, delta)
        self._parameter = np.array([invl, alpha],
                                   dtype=float)
        self.dt = dt
        self.v = v

    def get_parameter(self):
        return self._parameter

    def set_parameter(self, parameters):
        # parameters = parameters * 2.0 + self._parameter * -1
        invl, alpha = parameters
        self._parameter[:] = parameters
        A, B, C = get_discrete_matrices(self.dt, self.v, invl, alpha)
        self.system.transition_matrix[:] = A
        self.system.control_matrix[:] = B

    def get_coefficient(self, state=None, control=None):
        invl, alpha = self._parameter
        y, theta, phi = state
        u, = control

        J = np.zeros((3, 2))

        J[1, 0] = self.dt * self.v * phi
        J[2, 1] = -self.dt * phi + self.dt * u

        return J.T


def example():
    """Identify and control a 1st-order system using rls identificaiton."""
    v = 30
    dt = 0.02

    invl_true = 0.3
    alpha_true = 0.8

    invl_est = 0.8
    alpha_est = 1.0

    t_end = 500
    t = np.arange(0, t_end, dt)
    target_1 = 0.1 * np.sin(2 * np.pi * 0.2 * t)
    horizon = 50
    x0 = np.zeros([3])

    sys_true = mpc.LtiSystem(*get_discrete_matrices(
        dt, v, invl_true, alpha_true))
    sys_est = mpc.LtiSystem(*get_discrete_matrices(
        dt, v, invl_est, alpha_est))

    output_weighting = np.stack([1 * np.diag([1, 0])]*horizon)
    control_weighting = np.stack([10 * np.eye(1)]*horizon)
    lb = np.array([[-0.01]] * horizon)
    ub = np.array([[0.01]] * horizon)
    controller = mpc.Mpc(sys_est, horizon, output_weighting,
                         control_weighting=control_weighting)
    controller.set_control_limit(lb, ub)

    plant_true = mpc.LoggedPlant(sys_true, x0)

    forgetting_factor = 0.992
    rls_estimator = RlsEstimator(sys_est, 1000., dt, v, invl_est, alpha_est)

    q = np.diag([1., 1., 1.]) * dt ** 2 * 1.0
    r = np.diag([100., 1.]) * 10.0
    p0 = np.diag([0.01, 0.01, 0.01])*1.
    kf = mpc.Ekf(sys_est, x0, p0)

    x = x0
    ident_res = []
    x_kf_res = []
    for i in range(len(t) - horizon):
        target_output = target_1[i:i+horizon].reshape(horizon, 1)
        target_output = np.concatenate([target_output,
                                        np.zeros([horizon, 1])], 1)
        us = controller.solve(target_output, x)
        u = us[0]

        y_true = plant_true.step(u)
        # x_true = plant_true.get_state_history()[-1]

        # State estimation
        kf.predict(u, q)
        kf.update(y_true + np.random.randn(2)*[0.005, 0.0005], r)
        x_kf = kf.x

        # Model identification
        x_esti = sys_est.get_state(x, us[:1])[0]
        error = x_kf - x_esti
        rls_estimator.step(error, forgetting_factor, x, u)
        if i % 50 == 0:
            rls_estimator.update()

        x = x_kf

        ident_res.append(np.array(rls_estimator.get_parameter()))
        x_kf_res.append(x_kf)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(t, target_1, 'k-', label='target')
    ax.plot(t[:-horizon], plant_true.get_output_history()[:, 0],
            'r--', label='w/ control')
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('$y$ (m)')
    ax.legend()
    fig.tight_layout(pad=0.1)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(t[:-horizon], np.array(ident_res)[:, 0], 'k-', label='invl')
    ax.plot(t[:-horizon], np.array(ident_res)[:, 1], 'r-', label='alpha')
    ax.hlines(invl_true, t[0], t[-horizon], color='k', ls=':')
    ax.hlines(alpha_true, t[0], t[-horizon], color='r', ls=':')
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('Parameter values')
    ax.legend()
    fig.tight_layout(pad=0.1)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(t[:-horizon], 0.01 * np.array(x_kf_res)[:, 0],
            'k-', label='0.01$y$')
    ax.plot(t[:-horizon], 0.1 * np.array(x_kf_res)[:, 1],
            'r-', label=r'0.1$\theta$')
    ax.plot(t[:-horizon], np.array(x_kf_res)[:, 2],
            'b-', label=r'$\phi$')
    ax.plot(t[:-horizon], 0.01 * plant_true.get_state_history()[:, 0],
            'k:')
    ax.plot(t[:-horizon], 0.1 * plant_true.get_state_history()[:, 1],
            'r:')
    ax.plot(t[:-horizon], plant_true.get_state_history()[:, 2],
            'b:')
    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel('State values')
    ax.legend()
    fig.tight_layout(pad=0.1)


if __name__ == '__main__':
    example()
    plt.show()
