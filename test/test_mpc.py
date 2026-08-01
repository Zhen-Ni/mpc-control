#!/usr/bin/env python3

import unittest

import numpy as np
import mpc_control as mpc


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
        output_weighting = np.stack([np.eye(s.n_output)] * horizon)
        control_weighting = np.stack([np.eye(s.n_control)] * horizon)
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
        output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
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

        s = mpc.HomogeneousSystem(2, 1, 1, A, B, C)
        x0 = np.array([1., 0.])
        horizon = 50
        n_sim = 100
        uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        # Maintain the target output to 1.
        target_output = np.ones([horizon, s.n_output])
        output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
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

    def test_nonlinear_control_with_state_disturbance(self):
        """Maintain the output of a nonlinear system with disturbance."""
        def A(x, u):
            return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

        def B(x, u):
            return np.array([0., 1 + u[0]]).reshape(2, 1)

        def w(x, u):
            return np.array([0.1 + 0.1 * u[0], 0])

        def C(x):
            return np.array([1., 0.]).reshape(1, 2)

        s = mpc.NonlinearSystem(2, 1, 1, A, B, w, C,
                                lambda x: np.zeros([1]))
        x0 = np.array([1., 0.])
        horizon = 50
        n_sim = 100
        uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        # Maintain the target output to 1.
        target_output = np.ones([horizon, s.n_output])
        output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
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

        self.assertAlmostEqual(uncontrolled_output[-1, 0], 0.2)
        self.assertFalse(np.allclose(predicted_output[-5:], 1.))
        self.assertTrue(np.allclose(y[-10:], 1.))

    def test_nonlinear_control_with_output_disturbance(self):
        """Maintain the output of a nonlinear system with disturbance."""
        def A(x, u):
            return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

        def B(x, u):
            return np.array([0., 1 + u[0]]).reshape(2, 1)

        def w(x, u):
            return np.array([0.1 + 0.1 * u[0], 0])

        def C(x):
            return np.array([1., 0.]).reshape(1, 2)

        s = mpc.NonlinearSystem(2, 1, 1, A, B, w, C,
                                lambda x: np.array([-1.]))
        x0 = np.array([1., 0.])
        horizon = 50
        n_sim = 100
        uncontrolled_state = s.get_state(x0, np.zeros([n_sim, 1]))
        uncontrolled_output = s.get_output(uncontrolled_state)
        # Maintain the target output to 0.
        target_output = np.zeros([horizon, s.n_output])
        output_weighting_matrix = np.stack([np.eye(s.n_output)] * horizon)
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

        self.assertAlmostEqual(uncontrolled_output[-1, 0], 0.2-1.0)
        self.assertFalse(np.allclose(predicted_output[-5:], 0.))
        self.assertTrue(np.allclose(y[-10:], 0.))

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
        output_weighting = np.stack([np.eye(s.n_output)] * horizon)
        control_weighting = np.stack([np.eye(s.n_control)] * horizon)
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

    def test_constrained_output(self):
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
        output_weighting = np.stack([np.eye(s.n_output)] * horizon)
        control_weighting = np.stack([np.eye(s.n_control)] * horizon)
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
        output_weighting = np.stack([np.eye(s.n_output)] * horizon)
        control_weighting = np.stack([np.eye(s.n_control)] * horizon)
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

    def test_constrained_input_with_transform(self):
        """MPC control with constrained control input."""
        A = np.zeros([2, 2])
        B = np.eye(2)
        C = np.eye(2)
        s = mpc.LtiSystem(A, B, C)
        x0 = np.array([0., 0.])  # Initial state
        n_sim = 100             # Number of simulation steps
        horizon = 50            # MPC horizon
        bnd = 1                # bound of control input
        t = np.linspace(0, 2, n_sim+horizon)
        target_output = np.stack([t * np.cos(10 * t),
                                  t * np.sin(10 * t)]).T
        output_weighting = np.stack([np.eye(s.n_output)] * horizon)
        controller = mpc.Mpc(s, horizon,
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

        # Check whether the constraint is valid.
        self.assertTrue((np.array(u_cons)[:, 0] < 1e-3).all())
        self.assertTrue((np.array(u_cons)[:, 1] < 1 + 1e-3).all())

    def test_terminal_weighting(self):
        """Test trajectory planning using terminal output weighting."""
        # Use a double integrator system
        A = np.array([1., 1., 0., 1.]).reshape(2, 2)
        B = np.array([0., 1.]).reshape(2, 1)
        C = np.array([1., 0.]).reshape(1, 2)
        s = mpc.LtiSystem(A, B, C)

        x0 = np.array([0., 0.])
        horizon = 50
        target_output = np.ones([horizon, s.n_output])
        u0 = np.zeros([1])

        # 1. Terminal weighting: output weight is 0 for the
        # first N-1 steps, and 100 for the last step
        terminal_ow = np.zeros([horizon, s.n_output, s.n_output])
        terminal_ow[-1] = np.eye(s.n_output) * 100.0
        cw = np.stack([np.eye(s.n_control) * 0.1] * horizon)

        controller_terminal = mpc.Mpc(s, horizon,
                                      terminal_ow,
                                      cw)
        u_pred_terminal = controller_terminal.solve(target_output, x0, u0)
        y_pred_terminal = s.get_output(s.get_state(x0, u_pred_terminal))

        # 2. Uniform weighting: all steps have an output weight
        # of 100
        uniform_ow = np.stack([np.eye(s.n_output) * 100.0] * horizon)
        controller_uniform = mpc.Mpc(s, horizon,
                                     uniform_ow,
                                     cw)
        u_pred_uniform = controller_uniform.solve(target_output, x0, u0)
        y_pred_uniform = s.get_output(s.get_state(x0, u_pred_uniform))

        # Assertion 1: With terminal weighting, the system indeed
        # reaches the target at the last step
        self.assertAlmostEqual(y_pred_terminal[-1, 0], 1.0, delta=1e-3)

        # Assertion 2: With terminal weighting, since there is no
        # tracking penalty in the early stages, the early output of
        # the system should not tightly follow the target as with
        # uniform weighting (i.e., a larger deviation is
        # allowed). Under uniform weighting, the first step attempts
        # to approach the target, whereas under terminal weighting,
        # the early stages are slower.
        self.assertTrue(y_pred_terminal[10, 0] < y_pred_uniform[10, 0])

        # Assertion 3: Due to the lack of early output tracking
        # pressure, the control energy used by terminal weighting
        # should be less than (or equal to) that required by uniform
        # weighting to tightly follow the target.
        energy_terminal = np.sum(u_pred_terminal**2)
        energy_uniform = np.sum(u_pred_uniform**2)
        self.assertTrue(energy_terminal < energy_uniform)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
