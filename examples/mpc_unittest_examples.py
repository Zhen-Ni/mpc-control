#!/usr/bin/env python3

import numpy as np

from mpc_control import Mpc, LtiSystem, HomogeneousSystem, NonlinearSystem

import matplotlib.pyplot as plt


def example_1():
    """Control the system output to follow given path (sin wave)."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([0., 0.])
    n_sim = 100
    horizon = 50
    uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.sin(2*np.pi * 0.02 * np.arange(n_sim+horizon))
    output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
    # We do not use control weighting so that target output should
    # be tracked precisely.
    controller = Mpc(s, horizon,
                     output_weighting_matrix)

    y = np.zeros(n_sim)
    u = np.zeros([n_sim, s.n_control])
    xi = x0
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y[i] = s.get_output(xi.reshape(1, -1))[0, 0]
        u[i] = ui[0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(target_output[:n_sim], '--', label='target')
    ax.plot(y, label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u, label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_2():
    """Stablize a LTI system using MPC."""
    A = np.array([1.1, 1., 0., 0.9]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    n_sim = 100
    horizon = 50
    uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.zeros([horizon, s.n_output])
    output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
    control_weighting_matrix = np.stack([np.eye(s.n_control)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix)

    y = []
    u = []
    xi = x0
    for i in range(n_sim):
        ui = controller.solve(target_output, xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(y, label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u, label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_3():
    """Maintain the output of a nonlinear system."""
    def A(x, u):
        return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

    def B(x, u):
        return np.array([0., 1 + u[0]]).reshape(2, 1)

    def C(x):
        return np.array([1., 0.]).reshape(1, 2)

    s = HomogeneousSystem(2, 1, 1, A, B, C, lambda x: np.zeros([1]))
    x0 = np.array([1., 0.])
    horizon = 50
    n_sim = 100
    uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.ones([horizon, s.n_output])
    output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting_matrix)
    predicted_control = controller.solve(
        target_output, x0,
        state_ref=np.stack([x0] * horizon),
        control_ref=np.zeros([horizon, 1]))
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = np.concat([[x0], s.get_state(x0, ui[:-1])])
    y = np.zeros([n_sim])
    u = np.zeros([n_sim, 1])
    for i in range(n_sim):
        xi0 = xi[0]
        for j in range(20):
            ui = controller.solve(target_output, xi0,
                                  state_ref=xi,
                                  control_ref=ui)
            xi = np.concat([[xi0], s.get_state(x0, ui[:-1])])
        xi = s.get_state(xi0, ui)
        y[i] = s.get_output(xi[:1])[0, 0]
        u[i] = ui[0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(predicted_output, '--', label='predicted @ step 0')
    ax.plot(y, label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(predicted_control, '--', label='predicted @ step 0')
    ax2.plot(u, label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_4():
    """Maintain the output of a nonlinear system with disturbance."""
    def A(x, u):
        return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

    def B(x, u):
        return np.array([0., 1 + u[0]]).reshape(2, 1)

    def w(x, u):
        return np.array([0.1 + 0.1 * u[0], 0])

    def C(x):
        return np.array([1., 0.]).reshape(1, 2)

    s = NonlinearSystem(2, 1, 1, A, B, w, C, lambda x: np.zeros([1]))
    x0 = np.array([1., 0.])
    horizon = 50
    n_sim = 100
    uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.ones([horizon, s.n_output])
    output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting_matrix)
    predicted_control = controller.solve(
        target_output, x0,
        state_ref=np.stack([x0] * horizon),
        control_ref=np.zeros([horizon, 1]))
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = np.concat([[x0], s.get_state(x0, ui[:-1])])
    y = np.zeros([n_sim])
    u = np.zeros([n_sim, 1])
    for i in range(n_sim):
        xi0 = xi[0]
        for j in range(20):
            ui = controller.solve(target_output, xi0,
                                  state_ref=xi,
                                  control_ref=ui)
            xi = np.concat([[xi0], s.get_state(x0, ui[:-1])])
        xi = s.get_state(xi0, ui)
        y[i] = s.get_output(xi[:1])[0, 0]
        u[i] = ui[0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(predicted_output, '--', label='predicted @ step 0')
    ax.plot(y, label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(predicted_control, '--', label='predicted @ step 0')
    ax2.plot(u, label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_5():
    """Maintain the output of a nonlinear system with output disturbance."""
    def A(x, u):
        return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

    def B(x, u):
        return np.array([0., 1 + u[0]]).reshape(2, 1)

    def w(x, u):
        return np.array([0.1 + 0.1 * u[0], 0])

    def C(x):
        return np.array([1., 0.]).reshape(1, 2)

    s = NonlinearSystem(2, 1, 1, A, B, w, C,
                        lambda x: np.array([-1.]))
    x0 = np.array([1., 0.])
    horizon = 50
    n_sim = 100
    uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    # Maintain the target output to 0.
    target_output = np.zeros([horizon, s.n_output])
    output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting_matrix)
    predicted_control = controller.solve(
        target_output, x0,
        state_ref=np.stack([x0] * horizon),
        control_ref=np.zeros([horizon, 1]))
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = np.concat([[x0], s.get_state(x0, ui[:-1])])
    y = np.zeros([n_sim])
    u = np.zeros([n_sim, 1])
    for i in range(n_sim):
        xi0 = xi[0]
        for j in range(20):
            ui = controller.solve(target_output, xi0,
                                  state_ref=xi,
                                  control_ref=ui)
            xi = np.concat([[xi0], s.get_state(x0, ui[:-1])])
        xi = s.get_state(xi0, ui)
        y[i] = s.get_output(xi[:1])[0, 0]
        u[i] = ui[0]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(predicted_output, '--', label='predicted @ step 0')
    ax.plot(y, label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(predicted_control, '--', label='predicted @ step 0')
    ax2.plot(u, label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_6():
    """MPC control with constrained control input."""
    A = np.array([0.5, 1., 0., 0.9]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    n_sim = 100
    horizon = 50
    lb = -0.1
    ub = 0.1
    target_output = np.ones([n_sim+horizon, s.n_output])
    output_weighting = np.stack([np.eye(s.n_output)] * horizon)
    control_weighting = np.stack([np.eye(s.n_control)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting,
                     control_weighting)

    # Unconstrained
    xi = x0
    y_free = []
    u_free = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    # Constrained
    controller.set_control_limit(lb * np.ones([horizon, 1]),
                                 ub * np.ones([horizon, 1]))
    xi = x0
    y_cons = []
    u_cons = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_cons.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_cons.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(y_free, label='free')
    ax.plot(y_cons, label='constraint')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u_free, label='free')
    ax2.plot(u_cons, label='constraint')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_7():
    """MPC control with constrained control output."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([0., 0.])
    n_sim = 100
    horizon = 50
    lb = -0.8
    ub = 0.8
    target_output = np.ones([n_sim+horizon, s.n_output])
    output_weighting = np.stack([np.eye(s.n_output)] * horizon)
    control_weighting = np.stack([np.eye(s.n_control)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting,
                     control_weighting)

    # Unconstrained
    xi = x0
    y_free = []
    u_free = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    # Constrained
    controller.set_output_limit(lb * np.ones([horizon, 1]),
                                ub * np.ones([horizon, 1]))
    xi = x0
    y_cons = []
    u_cons = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_cons.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_cons.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(y_free, label='free')
    ax.plot(y_cons, label='constraint')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u_free, label='free')
    ax2.plot(u_cons, label='constraint')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_8():
    """MPC control with constrainted control change rate."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([0., 0.])
    n_sim = 100
    horizon = 50
    lb = -0.01
    ub = 0.01
    target_output = np.ones([n_sim+horizon, s.n_output])
    output_weighting = np.stack([np.eye(s.n_output)] * horizon)
    control_weighting = np.stack([np.eye(s.n_control)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting,
                     control_weighting)

    # Unconstrained
    xi = x0
    y_free = []
    u_free = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    # Constrained
    controller.set_control_rate_limit(lb * np.ones([horizon, 1]),
                                      ub * np.ones([horizon, 1]))
    ui = np.array([[0.]])
    xi = x0
    y_cons = []
    u_cons = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi, ui[0])
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_cons.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_cons.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(y_free, label='free')
    ax.plot(y_cons, label='constraint')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u_free, label='free')
    ax2.plot(u_cons, label='constraint')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_9():
    """MPC control with constrained control input with transform."""
    A = np.zeros([2, 2])
    B = np.eye(2)
    C = np.eye(2)
    s = LtiSystem(A, B, C)
    x0 = np.array([0., 0.])
    n_sim = 100
    horizon = 50
    bnd = 1
    t = np.linspace(0, 2, n_sim+horizon)
    target_output = np.stack([t * np.cos(10 * t),
                              t * np.sin(10 * t)]).T
    output_weighting = np.stack([np.eye(s.n_output)] * horizon)
    controller = Mpc(s, horizon,
                     output_weighting)
    bound = np.zeros([horizon, 2])
    bound[:, 0] = 1.
    trans = np.array([0, 1, 1, 0]).reshape(2, 2)
    trans = np.array([trans] * horizon)
    controller.set_control_limit(-bnd * bound,
                                 bnd * bound,
                                 trans)
    xi = x0
    y_cons = []
    u_cons = []
    for i in range(n_sim):
        ui = controller.solve(target_output[i: i+horizon], xi)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_cons.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_cons.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(np.array(y_cons)[:, 0], label='y1')
    ax.plot(np.array(y_cons)[:, 1], label='y2')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(np.array(u_cons)[:, 0], label='u1')
    ax2.plot(np.array(u_cons)[:, 1], label='u2')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


def example_10():
    """Test trajectory planning using terminal output weighting."""
    # Use a double integrator system
    A = np.array([1., 1., 0., 1.]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)

    x0 = np.array([0., 0.])
    horizon = 50
    target_output = np.ones([horizon, s.n_output])
    u0 = np.zeros([1])

    # 1. Terminal weighting: output weight is 0 for the
    # first N-1 steps, and 100 for the last step
    terminal_ow = np.zeros([horizon, s.n_output, s.n_output])
    terminal_ow[-1] = np.eye(s.n_output) * 100.0
    cw = np.stack([np.eye(s.n_control) * 0.1] * horizon)

    controller_terminal = Mpc(s, horizon,
                              terminal_ow,
                              cw)
    u_pred_terminal = controller_terminal.solve(target_output, x0, u0)
    y_pred_terminal = s.get_output(s.get_state(x0, u_pred_terminal))

    # 2. Uniform weighting: all steps have an output weight
    # of 100
    uniform_ow = np.stack([np.eye(s.n_output) * 100.0] * horizon)
    controller_uniform = Mpc(s, horizon,
                             uniform_ow,
                             cw)
    u_pred_uniform = controller_uniform.solve(target_output, x0, u0)
    y_pred_uniform = s.get_output(s.get_state(x0, u_pred_uniform))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(y_pred_terminal, '--', label='terminal weighting')
    ax.plot(y_pred_uniform, label='uniform weighting')
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(u_pred_terminal, '--', label='terminal weighting')
    ax2.plot(u_pred_uniform, label='uniform weighting')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')

    plt.tight_layout(pad=0.1)


if __name__ == '__main__':
    example_1()
    example_2()
    example_3()
    example_4()
    example_5()
    example_6()
    example_7()
    example_8()
    example_9()
    example_10()
    plt.show()
