#!/usr/bin/env python3

import unittest
import numpy as np
import mpc


class FirstOrder(mpc.LtiSystem, mpc.SupportRls):
    def __init__(self, tau, delta):
        A = np.array([[1-tau]])
        B = np.array([[tau]])
        C = np.array([[1.]])
        self._parameter = np.array([tau], dtype=float)
        mpc.LtiSystem.__init__(self, A, B, C)
        mpc.SupportRls.__init__(self, delta)

    def get_coefficient(self, state, control):
        result = np.array([[(-state[0] + control[0])]])
        return result

    def get_parameter(self):
        return self._parameter

    def set_parameter(self, parameters):
        self._parameter[:] = parameters
        tau = parameters[0]
        self.transition_matrix[0, 0] = 1 - tau
        self.control_matrix[0, 0] = tau  # type: ignore[misc]


class MultipleParameters(mpc.LtiSystem, mpc.SupportRls):
    def __init__(self, k1, k2, k3, delta):
        A = np.array([[k1, k2],
                      [0, 1-k3]])
        B = np.array([[0],
                      [k3]])
        C = np.array([[1., 0.],
                      [0., 1.]])
        self._parameter = np.array([k1, k2, k3], dtype=float)
        mpc.LtiSystem.__init__(self, A, B, C)
        mpc.SupportRls.__init__(self, delta)

    def get_coefficient(self, state, control):
        x0, x1 = state
        u, = control
        result = np.array([[x0, x1, 0],
                           [0, 0, -x1+u]])
        return result.T

    def get_parameter(self):
        return self._parameter

    def set_parameter(self, parameters):
        self._parameter[:] = parameters
        k1, k2, k3 = parameters
        self.transition_matrix[:] = np.array([[k1, k2],
                                              [0, 1-k3]])
        self.control_matrix[:] = np.array([[0],
                                           [k3]])


class NonlinearSystem(mpc.LtiSystem, mpc.SupportRls):
    def __init__(self, k1, k2, k3, delta):
        A = np.array([[k1, np.tan(k2)],
                      [0, 1-k3]])
        B = np.array([[0],
                      [k3]])
        C = np.array([[1., 0.],
                      [0., 1.]])
        self._parameter = np.array([k1, k2, k3], dtype=float)
        mpc.LtiSystem.__init__(self, A, B, C)
        mpc.SupportRls.__init__(self, delta)

    def get_coefficient(self, state, control):
        x0, x1 = state
        u, = control
        k2 = self._parameter[1]
        result = np.array([[x0, x1 / np.cos(k2)**2, 0],
                           [0, 0, -x1+u]])
        return result.T

    def get_parameter(self):
        return self._parameter

    def set_parameter(self, parameters):
        self._parameter[:] = parameters
        k1, k2, k3 = parameters
        self.transition_matrix[:] = np.array([[k1, np.tan(k2)],
                                              [0, 1-k3]])
        self.control_matrix[:] = np.array([[0],
                                           [k3]])


class TestRls(unittest.TestCase):
    def test_first_order(self):
        """Test parameter identification of a first-order system."""
        tau = 0.2
        delta = 100
        s_ref = FirstOrder(tau, delta)
        s_rls = FirstOrder(0.0, delta)

        n = 100
        t = 20
        state = np.array([0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = s_ref.get_state(state, u[None])[0]
            y_ref = s_ref.get_output(x_ref[None])[0]
            x_rls = s_rls.get_state(state, u[None])[0]
            y_rls = s_ref.get_output(x_rls[None])[0]
            e = y_rls - y_ref
            s_rls.update(e, forgetting_factor, x_ref, u)
            state = x_ref
            convergence.append(s_rls.get_parameter()[0])

        self.assertAlmostEqual(convergence[-1], tau)

    def test_multiple_parameters(self):
        """Test identification a 2-DOF system with 3 unknown parameters."""
        k1, k2, k3 = 0.9, 1.0, 0.2
        delta = 100
        s_ref = MultipleParameters(k1, k2, k3, delta)
        s_rls = MultipleParameters(0.0, 0.0, 0.0, delta)

        n = 100
        t = 20
        state = np.array([0.0, 0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = s_ref.get_state(state, u[None])[0]
            y_ref = s_ref.get_output(x_ref[None])[0]
            x_rls = s_rls.get_state(state, u[None])[0]
            y_rls = s_ref.get_output(x_rls[None])[0]
            e = y_rls - y_ref
            s_rls.update(e, forgetting_factor, x_ref, u)
            state = x_ref
            convergence.append(s_rls.get_parameter())

        self.assertAlmostEqual(convergence[-1][0], k1, delta=0.01)
        self.assertAlmostEqual(convergence[-1][1], k2, delta=0.01)
        self.assertAlmostEqual(convergence[-1][2], k3, delta=0.01)

    def test_nonlinear_system(self):
        """Test identification a 2-DOF nonlinear system."""
        k1, k2, k3 = 0.9, 1.5, 0.2
        delta = 100
        s_ref = NonlinearSystem(k1, k2, k3, delta)
        s_rls = NonlinearSystem(0.0, 0.0, 0.0, delta)

        n = 100
        t = 20
        state = np.array([0.0, 0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = s_ref.get_state(state, u[None])[0]
            y_ref = s_ref.get_output(x_ref[None])[0]
            x_rls = s_rls.get_state(state, u[None])[0]
            y_rls = s_ref.get_output(x_rls[None])[0]
            e = y_rls - y_ref
            s_rls.update(e, forgetting_factor, x_ref, u)
            state = x_ref
            convergence.append(s_rls.get_parameter())

        self.assertAlmostEqual(convergence[-1][0], k1, delta=0.01)
        self.assertAlmostEqual(convergence[-1][1], k2, delta=0.01)
        self.assertAlmostEqual(convergence[-1][2], k3, delta=0.01)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
