#!/usr/bin/env python3

import unittest

import numpy as np
import mpc


class TestMpc(unittest.TestCase):
    """Unittest for the mpc controller."""

    def test_stablize_lti_system(self):
        """Stablize a LTI system using MPC."""
        A = np.array([1.1, 1., 0., 0.9]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([1., 0.])  # Initial state
        n_sim = 100             # Number of simulation steps
        horizon = 50            # MPC horizon
        uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        target_output = np.zeros([horizon, s.n_output])
        output_weighting = np.eye(s.n_output)
        control_weighting = np.eye(s.n_control)
        controller = mpc.Mpc(s, horizon,
                             output_weighting,
                             control_weighting)

        xi = x0
        y = []
        u = []
        for i in range(n_sim):
            ui = controller.solve(target_output, xi)
            xi = s.get_state(xi, ui[0:1]).reshape(-1)
            y.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
            u.append(ui[0].reshape(-1))

        # The uncontrolled system should diverge.
        self.assertAlmostEqual(uncontrolled_output[-1, 0], 1.1**n_sim)
        # The endpoints of the controlled system output should be
        # stablized.
        for i in range(n_sim - 10, n_sim):
            self.assertAlmostEqual(y[i][0], 0.0)

    def test_controlled_output(self):
        """Control the system output to follow given path."""
        A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([0., 0.])
        n_sim = 100
        horizon = 50
        uncontrolled_state = s.get_state(x0, np.zeros([horizon, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        target_output = np.sin(2*np.pi * 0.02 * np.arange(n_sim+horizon))
        output_weighting_matrix = np.eye(s.n_output)
        # We do not use control weighting so that target output should
        # be tracked precisely.
        controller = mpc.Mpc(s, horizon,
                             output_weighting_matrix)
        y = np.zeros(n_sim)
        xi = x0
        for i in range(n_sim):
            ui = controller.solve(target_output[i: i+horizon],
                                  xi)
            xi = s.get_state(xi, ui[0:1]).reshape(-1)
            y[i] = s.get_output(xi.reshape(1, -1))[0, 0]

        # The uncontrolled system should be 0
        self.assertAlmostEqual(uncontrolled_output[-1, 0], 0)
        # Check the output
        self.assertTrue(np.allclose(target_output[:n_sim], y, atol=1e-6))

    def test_nonlinear_control(self):
        """Maintain the output of a nonlinear system."""
        def A(x, u):
            return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

        def B(x, u):
            return np.array([0., 1 + u[0]]).reshape(2, 1)

        def C(x):
            return np.array([1., 0.]).reshape(1, 2)

        s = mpc.LinearJacSystem(2, 1, 1, A, B, C)
        x0 = np.array([1., 0.])
        horizon = 50
        n_sim = 100
        uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        # Maintain the target output to 1.
        target_output = np.ones([horizon, s.n_output])
        output_weighting_matrix = np.eye(s.n_output)
        controller = mpc.Mpc(s, horizon,
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
        for i in range(n_sim):
            xi0 = xi[0]
            for j in range(20):
                ui = controller.solve(target_output, xi0,
                                      state_ref=xi,
                                      control_ref=ui)
                xi = np.concat([[xi0], s.get_state(x0, ui[:-1])])
            xi = s.get_state(xi0, ui)
            y[i] = s.get_output(xi[:1])[0, 0]

        self.assertAlmostEqual(uncontrolled_output[-1, 0], 0.)
        # Predicted output should never be precise without iteration
        # for linear system.
        self.assertFalse(np.allclose(predicted_output[-5:], 1.))
        self.assertTrue(np.allclose(y[-10:], 1.))

    def test_constrained_input(self):
        """MPC control with constrained control input."""
        A = np.array([0.5, 1., 0., 0.9]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([1., 0.])  # Initial state
        n_sim = 100             # Number of simulation steps
        horizon = 50            # MPC horizon
        lb = -0.1               # Lower bound of control input
        ub = 0.1                # Upper bound of control input
        target_output = np.ones([n_sim+horizon, s.n_output])
        output_weighting = np.eye(s.n_output)
        control_weighting = np.eye(s.n_control)
        controller = mpc.Mpc(s, horizon,
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

        for i in range(n_sim - 10, n_sim):
            # Use a relatively large torlerance as control input is
            # also used as the cost.
            self.assertAlmostEqual(
                y_free[i][0], target_output[i, 0], delta=1e-2)

        # Check whether the constraint is valid.
        self.assertTrue((np.array(u_free) > ub + 1e-3).any())
        self.assertTrue((np.array(u_cons) < ub + 1e-3).all())

    def test_constrained_input(self):
        """MPC control with constrained control output."""
        A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([0., 0.])  # Initial state
        n_sim = 100             # Number of simulation steps
        horizon = 50            # MPC horizon
        lb = -0.8               # Lower bound of controlled output
        ub = 0.8                # Upper bound of controlled output
        target_output = np.ones([n_sim+horizon, s.n_output])
        output_weighting = np.eye(s.n_output)
        control_weighting = np.eye(s.n_control)
        controller = mpc.Mpc(s, horizon,
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

        # Check whether the constraint is valid.
        self.assertTrue((np.array(y_free) > ub + 1e-3).any())
        self.assertTrue((np.array(y_cons) < ub + 1e-3).all())

    def test_constrainted_control_rate(self):
        A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([0., 0.])  # Initial state
        n_sim = 100             # Number of simulation steps
        horizon = 50            # MPC horizon
        lb = -0.01              # Lower bound of controlled output
        ub = 0.01               # Upper bound of controlled output
        target_output = np.ones([n_sim+horizon, s.n_output])
        output_weighting = np.eye(s.n_output)
        control_weighting = np.eye(s.n_control)
        controller = mpc.Mpc(s, horizon,
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

        # Check whether the constraint is valid.
        du_free = np.diff(np.array(u_free).reshape(-1))
        du_cons = np.diff(np.array(u_cons).reshape(-1))

        self.assertTrue((du_free > ub + 1e-3).any()
                        or (du_free < lb - 1e-3).any())
        self.assertTrue((du_cons < ub + 1e-3).all()
                        and (du_cons > lb - 1e-3).all())


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
