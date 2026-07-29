#!/usr/bin/env python3

import unittest
import numpy as np
from scipy.signal import cont2discrete

import mpc


class TestEKF(unittest.TestCase):
    def test_observe_state(self):
        """State observer of a linear system."""
        # A 2 DOF vibration system
        m1, m2 = 1.0, 1.5
        c1, c2 = 0.2, 0.3
        k1, k2 = 15.0, 20.0
        # state variables: [x1, v1, x2, v2]
        # inputs: [u1, -u2]
        A = np.array([
            [0, 1, 0, 0],
            [-(k1 + k2) / m1, -(c1 + c2) / m1, k2 / m1, c2 / m1],
            [0, 0, 0, 1],
            [k2 / m2, c2 / m2, -k2 / m2, -c2 / m2]
        ])
        B = np.array([
            [0],
            [1 / m1],
            [0],
            [-1 / m2]
        ])
        # Output the displacement of the second point
        C = np.array([
            [0, 0, 1, 0]
        ])

        # Discretize the continuous-time system with sampling time dt
        dt = 0.05
        D = np.zeros((C.shape[0], B.shape[1]))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt)

        system = mpc.LtiSystem(
            transition_matrix=A_d,
            control_matrix=B_d,
            output_matrix=C_d
        )

        t_end = 10.0
        t = np.arange(0, t_end, dt)
        inputs = np.sin(t)

        x0 = np.zeros([4])
        p0 = np.eye(4) * 100
        kf = mpc.Ekf(system, x0, p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[1.0]])
            kf.predict(u, p)
            x = system.get_state(x, u.reshape(1, 1))[0]
            y = system.get_output(x.reshape(1, 4))[0]
            r = np.array([[1.0]])
            kf.update(y, r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(np.allclose(x_true, x_pred))

    def test_observe_state_noised(self):
        """State observer of a linear system with noise."""
        m1, m2 = 1.0, 1.5
        c1, c2 = 0.2, 0.3
        k1, k2 = 15.0, 20.0
        # state variables: [x1, v1, x2, v2]
        # inputs: [u1, -u2]
        A = np.array([
            [0, 1, 0, 0],
            [-(k1 + k2) / m1, -(c1 + c2) / m1, k2 / m1, c2 / m1],
            [0, 0, 0, 1],
            [k2 / m2, c2 / m2, -k2 / m2, -c2 / m2]
        ])
        B = np.array([
            [0],
            [1 / m1],
            [0],
            [-1 / m2]
        ])
        # Output the displacement of the second point
        C = np.array([
            [0, 0, 1, 0]
        ])

        # Discretize the continuous-time system with sampling time dt
        dt = 0.05
        D = np.zeros((C.shape[0], B.shape[1]))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt)

        system = mpc.LtiSystem(
            transition_matrix=A_d,
            control_matrix=B_d,
            output_matrix=C_d
        )

        t_end = 10.0
        t = np.arange(0, t_end, dt)
        inputs = np.sin(t)

        x0 = np.zeros([4])
        p0 = np.eye(4) * 100
        kf = mpc.Ekf(system, x0, p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[0.1]])
            kf.predict(u+0.01*np.random.randn(1), p)
            x = system.get_state(x, u.reshape(1, 1))[0]
            y = system.get_output(x.reshape(1, 4))[0]
            r = np.array([[0.4]])
            kf.update(y+0.02*np.random.randn(1), r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(abs(((x_true - x_pred)[-100:]) < 0.1).all())

    def test_observe_nonlinear_state(self):
        """State observer of a nonlinear system."""
        def A_true(x, u):
            return np.array([0.5, 1.,
                             0., 0.48 + np.sin(u[0]+0.02)]).reshape(2, 2)

        def A_est(x, u):
            return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

        def B(x, u):
            return np.array([0., 1 + u[0]]).reshape(2, 1)

        def C(x):
            return np.array([1., 0.]).reshape(1, 2)

        system_true = mpc.HomogeneousSystem(2, 1, 1, A_true, B, C,
                                            lambda x: np.zeros([1]))
        system_est = mpc.HomogeneousSystem(2, 1, 1, A_est, B, C,
                                           lambda x: np.zeros([1]))

        n = 100
        inputs = np.sin(2 * np.pi * np.arange(n) / 20)

        x0 = np.zeros([2])
        p0 = np.eye(2) * 1.0
        kf = mpc.Ekf(system_est, np.random.randn(2), p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[1.0]])
            kf.predict(u, p)
            x = system_true.get_state(x, u.reshape(1, 1))[0]
            y = system_true.get_output(x.reshape(1, 2))[0]
            r = np.array([[1.0]])
            kf.update(y, r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(np.allclose(x_true[-10:], x_pred[-10:], atol=0.1))


class TestUKF(unittest.TestCase):
    def test_observe_state(self):
        """State observer of a linear system."""
        # A 2 DOF vibration system
        m1, m2 = 1.0, 1.5
        c1, c2 = 0.2, 0.3
        k1, k2 = 15.0, 20.0
        # state variables: [x1, v1, x2, v2]
        # inputs: [u1, u2]
        A = np.array([
            [0, 1, 0, 0],
            [-(k1 + k2) / m1, -(c1 + c2) / m1, k2 / m1, c2 / m1],
            [0, 0, 0, 1],
            [k2 / m2, c2 / m2, -k2 / m2, -c2 / m2]
        ])
        B = np.array([
            [0],
            [1 / m1],
            [0],
            [-1 / m2]
        ])
        # Output the displacement of the second point
        C = np.array([
            [0, 0, 1, 0]
        ])

        # Discretize the continuous-time system with sampling time dt
        dt = 0.05
        D = np.zeros((C.shape[0], B.shape[1]))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt)

        system = mpc.LtiSystem(
            transition_matrix=A_d,
            control_matrix=B_d,
            output_matrix=C_d
        )

        t_end = 10.0
        t = np.arange(0, t_end, dt)
        inputs = np.sin(t)

        x0 = np.zeros([4])
        p0 = np.eye(4) * 100
        kf = mpc.Ukf(system, x0, p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[1.0]])
            kf.predict(u, p)
            x = system.get_state(x, u.reshape(1, 1))[0]
            y = system.get_output(x.reshape(1, 4))[0]
            r = np.array([[1.0]])
            kf.update(y, r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(np.allclose(x_true, x_pred))

    def test_observe_state_noised(self):
        """State observer of a linear system with noise."""
        m1, m2 = 1.0, 1.5
        c1, c2 = 0.2, 0.3
        k1, k2 = 15.0, 20.0
        # state variables: [x1, v1, x2, v2]
        # inputs: [u1, -u2]
        A = np.array([
            [0, 1, 0, 0],
            [-(k1 + k2) / m1, -(c1 + c2) / m1, k2 / m1, c2 / m1],
            [0, 0, 0, 1],
            [k2 / m2, c2 / m2, -k2 / m2, -c2 / m2]
        ])
        B = np.array([
            [0],
            [1 / m1],
            [0],
            [-1 / m2]
        ])
        # Output the displacement of the second point
        C = np.array([
            [0, 0, 1, 0]
        ])

        # Discretize the continuous-time system with sampling time dt
        dt = 0.05
        D = np.zeros((C.shape[0], B.shape[1]))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A, B, C, D), dt)

        system = mpc.LtiSystem(
            transition_matrix=A_d,
            control_matrix=B_d,
            output_matrix=C_d
        )

        t_end = 10.0
        t = np.arange(0, t_end, dt)
        inputs = np.sin(t)

        x0 = np.zeros([4])
        p0 = np.eye(4) * 100
        kf = mpc.Ukf(system, x0, p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[0.1]])
            kf.predict(u+0.01*np.random.randn(1), p)
            x = system.get_state(x, u.reshape(1, 1))[0]
            y = system.get_output(x.reshape(1, 4))[0]
            r = np.array([[0.4]])
            kf.update(y+0.02*np.random.randn(1), r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(abs(((x_true - x_pred)[-100:]) < 0.1).all())

    def test_observe_nonlinear_state(self):
        """State observer of a nonlinear system."""
        def A_true(x, u):
            return np.array([0.5, 1., 0.,
                             0.48 + np.sin(u[0]+0.02)]).reshape(2, 2)

        def A_est(x, u):
            return np.array([0.5, 1., 0., 0.5 + np.sin(u[0])]).reshape(2, 2)

        def B(x, u):
            return np.array([0., 1 + u[0]]).reshape(2, 1)

        def C(x):
            return np.array([1., 0.]).reshape(1, 2)

        system_true = mpc.HomogeneousSystem(2, 1, 1, A_true, B, C,
                                            lambda x: np.zeros([1]))
        system_est = mpc.HomogeneousSystem(2, 1, 1, A_est, B, C,
                                           lambda x: np.zeros([1]))

        n = 100
        inputs = np.sin(2 * np.pi * np.arange(n) / 20)

        x0 = np.zeros([2])
        p0 = np.eye(2) * 1.0
        kf = mpc.Ukf(system_est, np.random.randn(2), p0)

        x = x0.copy()

        x_true_list = []
        x_pred_list = []

        for i, u in enumerate(inputs):
            u = np.array([u])
            p = np.array([[1.0]])
            kf.predict(u, p)
            x = system_true.get_state(x, u.reshape(1, 1))[0]
            y = system_true.get_output(x.reshape(1, 2))[0]
            r = np.array([[1.0]])
            kf.update(y, r)

            x_true_list.append(x)
            x_pred_list.append(kf.x)

        x_true = np.array(x_true_list)
        x_pred = np.array(x_pred_list)

        self.assertTrue(np.allclose(x_true[-10:], x_pred[-10:], atol=0.1))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
