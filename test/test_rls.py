#!/usr/bin/env python3

import unittest
import numpy as np
import mpc_control as mpc


class FirstOrderRls(mpc.Rls):
    def __init__(self,
                 system: mpc.LtiSystem,
                 delta: float,
                 initial_tau: float):
        self._parameter = np.array([initial_tau], dtype=float)
        super().__init__(system, delta)

    def get_coefficient(self, state, control):
        result = np.array([[(-state[0] + control[0])]])
        return result

    def get_parameter(self):
        return self._parameter

    def set_parameter(self, parameters):
        self._parameter[:] = parameters
        tau = parameters[0]
        self._system.transition_matrix[0, 0] = 1 - tau
        self._system.control_matrix[0, 0] = tau


class MultipleParametersRls(mpc.Rls):
    def __init__(self,
                 system: mpc.LtiSystem,
                 delta: float,
                 initial_params: np.ndarray):
        self._parameter = initial_params.copy()
        super().__init__(system, delta)

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
        self._system.transition_matrix[:] = np.array([[k1, k2],
                                                      [0, 1-k3]])
        self._system.control_matrix[:] = np.array([[0],
                                                   [k3]])


class NonlinearRls(mpc.Rls):
    def __init__(self,
                 system: mpc.LtiSystem,
                 delta: float,
                 initial_params: np.ndarray):
        self._parameter = initial_params.copy()
        super().__init__(system, delta)

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
        self._system.transition_matrix[:] = np.array([[k1, np.tan(k2)],
                                                      [0, 1-k3]])
        self._system.control_matrix[:] = np.array([[0],
                                                   [k3]])


class TestRls(unittest.TestCase):
    def test_first_order(self):
        """Test parameter identification of a first-order system."""
        tau = 0.2
        delta = 100
        sys_ref = mpc.LtiSystem(
            transition_matrix=np.array([[1-tau]]),
            control_matrix=np.array([[tau]]),
            output_matrix=np.array([[1.]]))
        sys_rls = mpc.LtiSystem(
            transition_matrix=np.array([[1.0]]),
            control_matrix=np.array([[0.0]]),
            output_matrix=np.array([[1.]]))
        s_rls = FirstOrderRls(sys_rls, delta, 0.0)

        n = 100
        t = 20
        state = np.array([0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = sys_ref.get_state(state, u[None])[0]
            y_ref = sys_ref.get_output(x_ref[None])[0]
            x_rls = sys_rls.get_state(state, u[None])[0]
            y_rls = sys_ref.get_output(x_rls[None])[0]
            e = y_ref - y_rls
            s_rls.step(e, forgetting_factor, x_ref, u)
            s_rls.update()
            state = x_ref
            convergence.append(s_rls.get_parameter()[0])

        self.assertAlmostEqual(convergence[-1], tau)

    def test_multiple_parameters(self):
        """Test identification a 2-DOF system with 3 unknown parameters."""
        k1, k2, k3 = 0.9, 1.0, 0.2
        delta = 100
        sys_ref = mpc.LtiSystem(
            transition_matrix=np.array([[k1, k2],
                                        [0, 1-k3]]),
            control_matrix=np.array([[0],
                                     [k3]]),
            output_matrix=np.array([[1., 0.],
                                    [0., 1.]]))
        sys_rls = mpc.LtiSystem(
            transition_matrix=np.array([[0.0, 0.0],
                                        [0, 1.0]]),
            control_matrix=np.array([[0],
                                     [0.0]]),
            output_matrix=np.array([[1., 0.],
                                    [0., 1.]]))
        s_rls = MultipleParametersRls(
            sys_rls, delta, np.array([0.0, 0.0, 0.0], dtype=float))

        n = 100
        t = 20
        state = np.array([0.0, 0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = sys_ref.get_state(state, u[None])[0]
            y_ref = sys_ref.get_output(x_ref[None])[0]
            x_rls = sys_rls.get_state(state, u[None])[0]
            y_rls = sys_ref.get_output(x_rls[None])[0]
            e = y_ref - y_rls
            s_rls.step(e, forgetting_factor, x_ref, u)
            s_rls.update()
            state = x_ref
            convergence.append(s_rls.get_parameter())

        self.assertAlmostEqual(convergence[-1][0], k1, delta=0.01)
        self.assertAlmostEqual(convergence[-1][1], k2, delta=0.01)
        self.assertAlmostEqual(convergence[-1][2], k3, delta=0.01)

    def test_nonlinear_system(self):
        """Test identification a 2-DOF nonlinear system."""
        k1, k2, k3 = 0.9, 1.5, 0.2
        delta = 100
        sys_ref = mpc.LtiSystem(
            transition_matrix=np.array([[k1, np.tan(k2)],
                                        [0, 1-k3]]),
            control_matrix=np.array([[0],
                                     [k3]]),
            output_matrix=np.array([[1., 0.],
                                    [0., 1.]]))
        sys_rls = mpc.LtiSystem(
            transition_matrix=np.array([[0.0, np.tan(0.0)],
                                        [0, 1.0]]),
            control_matrix=np.array([[0],
                                     [0.0]]),
            output_matrix=np.array([[1., 0.],
                                    [0., 1.]]))
        s_rls = NonlinearRls(
            sys_rls, delta, np.array([0.0, 0.0, 0.0], dtype=float))

        n = 100
        t = 20
        state = np.array([0.0, 0.0])
        controls = np.array(([1]*(t//2)+[0]*(t//2))*(n//t)).reshape(n, 1)

        convergence = []

        forgetting_factor = 0.9
        for i, u in enumerate(controls):
            x_ref = sys_ref.get_state(state, u[None])[0]
            y_ref = sys_ref.get_output(x_ref[None])[0]
            x_rls = sys_rls.get_state(state, u[None])[0]
            y_rls = sys_ref.get_output(x_rls[None])[0]
            e = y_ref - y_rls
            s_rls.step(e, forgetting_factor, x_ref, u)
            s_rls.update()
            state = x_ref
            convergence.append(s_rls.get_parameter())

        self.assertAlmostEqual(convergence[-1][0], k1, delta=0.01)
        self.assertAlmostEqual(convergence[-1][1], k2, delta=0.01)
        self.assertAlmostEqual(convergence[-1][2], k3, delta=0.01)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
