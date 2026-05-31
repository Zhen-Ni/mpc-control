#!/usr/bin/env python3

import numpy as np

from discrete import LtiSystem, NonlinearJacSystem
from mpc import Mpc

import matplotlib.pyplot as plt


def example_1():
    """Control a stable 2-DOF LTI system."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    u0 = np.zeros([1])
    horizon = 50
    uncontrolled_state = s.get_state(x0, np.zeros([horizon, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    control_delta_weighting_matrix = np.eye(s.n_control) * 0
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix,
                     control_delta_weighting_matrix)
    predicted_control = controller.solve(target_output, x0, u0)
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = s.get_state(x0, ui)
    y = []
    u = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi[0], ui[0])
        xi = s.get_state(xi[0], ui)
        y.append(s.get_output(xi[:1]).reshape(-1))
        u.append(ui[0].reshape(-1))

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


def example_2():
    """Stablize 2-DOF LTI system."""
    A = np.array([1.01, 1., 0., 0.9]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    u0 = np.zeros([1])
    n_sim = 100
    horizon = 50
    uncontrolled_state = s.get_state(x0, np.zeros([horizon, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.zeros([horizon, s.n_output])
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix)
    predicted_control = controller.solve(target_output, x0, u0)
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y = []
    u = []
    for i in range(n_sim):
        if i == 0:
            ui = controller.solve(target_output, xi, ui[0])
        else:
            ui = controller.solve(target_output, xi, ui[0], control_ref=ui)
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u.append(ui[0].reshape(-1))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(211)
    ax.plot(predicted_output, label='predicted @ step 0')
    ax.plot(y, '--', label='controlled')
    ax.plot(uncontrolled_output, ':', label='uncontrolled')
    ax.set_ylim(-0.1, 2)
    ax.legend()
    ax.grid()
    ax.set_ylabel('output')

    ax2 = fig.add_subplot(212, sharex=ax)
    ax2.plot(predicted_control, label='predicted @ step 0')
    ax2.plot(u, '--', label='controlled')
    ax2.legend()
    ax2.grid()
    ax2.set_ylabel('control')


def example_3():
    """Control a nonlinear system."""
    def A(x, u): return np.array(
        [0.9, 1., 0., 0.5 + np.sin(x[1])]).reshape(2, 2)

    def B(x, u): return np.array([0., np.cos(u[0])]).reshape(2, 1)
    def C(x): return np.array([1., 0.]).reshape(1, 2)
    s = NonlinearJacSystem(2, 1, 1, A, B, C)
    x0 = np.array([1., 0.])
    u0 = np.zeros([1])
    horizon = 100
    uncontrolled_state = s.get_state(x0, np.zeros([horizon, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    control_delta_weighting_matrix = np.eye(s.n_control) * 0
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix,
                     control_delta_weighting_matrix)
    predicted_control = controller.solve(
        target_output, x0, u0,
        np.stack([x0] * horizon), np.zeros([horizon, 1]))
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = s.get_state(x0, ui)
    u = []
    for i in range(horizon):
        xi0 = xi[0]
        for j in range(20):
            ui = controller.solve(target_output, xi0, ui[0], xi, ui)
            xi = s.get_state(xi[0], ui)
        xi = s.get_state(xi0, ui)
        u.append(ui[0].reshape(-1))
    u = np.array(u).reshape(-1, 1)
    x = s.get_state(x0, u)
    y = s.get_output(x)

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
    """Control a stable 2-DOF LTI system with constrainted control input."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    horizon = 50
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    control_delta_weighting_matrix = np.eye(s.n_control) * 0
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix,
                     control_delta_weighting_matrix)

    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_free = []
    u_free = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    controller.set_control_limit(-0.08 * np.ones([horizon, 1]),
                                 0.08 * np.ones([horizon, 1]))
    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_cons = []
    u_cons = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
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


def example_5():
    """Control a stable 2-DOF LTI system with constrainted control output."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    horizon = 50
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    control_delta_weighting_matrix = np.eye(s.n_control) * 0
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix,
                     control_delta_weighting_matrix)

    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_free = []
    u_free = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    controller.set_output_limit(0.0 * np.ones([horizon, 1]),
                                0.9 * np.ones([horizon, 1]))
    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_cons = []
    u_cons = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
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


def example_6():
    """Control a stable 2-DOF LTI system with control rate weighting."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    u0 = np.zeros([1])
    horizon = 50
    uncontrolled_state = s.get_state(x0, np.zeros([horizon, 1]))
    uncontrolled_output = s.get_output(uncontrolled_state)
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control) * 0
    control_delta_weighting_matrix = np.eye(s.n_control) * 100
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix,
                     control_delta_weighting_matrix)
    predicted_control = controller.solve(target_output, x0, u0)
    predicted_state = s.get_state(x0, predicted_control)
    predicted_output = s.get_output(predicted_state)

    ui = np.zeros([horizon, 1])
    xi = s.get_state(x0, ui)
    y = []
    u = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi[0], ui[0])
        xi = s.get_state(xi[0], ui)
        y.append(s.get_output(xi[:1]).reshape(-1))
        u.append(ui[0].reshape(-1))

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


def example_7():
    """Control a stable 2-DOF LTI system with constrainted control change rate."""
    A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
    B = np.array([0., 1.]).reshape(2, 1)
    C = np.array([1., 0.]).reshape(1, 2)
    s = LtiSystem(A, B, C)
    x0 = np.array([1., 0.])
    horizon = 50
    target_output = np.zeros([horizon, s.n_output]) + 1.
    output_weighting_matrix = np.eye(s.n_output)
    control_weighting_matrix = np.eye(s.n_control)
    controller = Mpc(s, horizon,
                     output_weighting_matrix,
                     control_weighting_matrix)

    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_free = []
    u_free = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
        xi = s.get_state(xi, ui[0:1]).reshape(-1)
        y_free.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
        u_free.append(ui[0].reshape(-1))

    controller.set_control_rate_limit(-0.01 * np.ones([horizon, 1]),
                                      0.01 * np.ones([horizon, 1]))
    ui = np.zeros([1])
    xi = s.get_state(x0, ui.reshape(1, -1)).reshape(-1)
    y_cons = []
    u_cons = []
    for i in range(horizon):
        ui = controller.solve(target_output, xi, ui[0])
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
    
if __name__ == '__main__':
    example_1()
    example_2()
    example_3()
    example_4()
    example_5()
    example_6()
    example_7()
    plt.show()
