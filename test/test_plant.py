#!/usr/bin/env python3

"""Unittest for the simulation environment."""

import unittest
import numpy as np

from mpc_control.discrete import LtiSystem
from mpc_control.plant import Plant, LoggedPlant


class TestPlant(unittest.TestCase):
    """Unittest for the Plant class."""

    def test_step(self):
        """Test single step simulation of a simple system."""
        # A simple integrator system
        sys = LtiSystem(
            transition_matrix=np.array([[1.0]]),
            control_matrix=np.array([[1.0]]),
            output_matrix=np.array([[1.0]]))
        plant = Plant(sys, initial_state=np.array([0.0]))

        y = plant.step(np.array([1.0]))
        np.testing.assert_allclose(y, np.array([1.0]))

        y = plant.step(np.array([1.0]))
        np.testing.assert_allclose(y, np.array([2.0]))

    def test_default_initial_state(self):
        """Test default initial state when None is provided."""
        sys = LtiSystem(
            transition_matrix=np.array([[1.0, 0.0],
                                        [0.0, 1.0]]),
            control_matrix=np.array([[1.0],
                                     [0.0]]),
            output_matrix=np.array([[1.0, 0.0]]))
        plant = Plant(sys)

        y = plant.step(np.array([1.0]))
        np.testing.assert_allclose(y, np.array([1.0]))

    def test_invalid_initial_state_shape(self):
        """Test that invalid initial state shape raises ValueError."""
        sys = LtiSystem(
            transition_matrix=np.array([[1.0]]),
            control_matrix=np.array([[1.0]]),
            output_matrix=np.array([[1.0]]))
        with self.assertRaises(ValueError):
            Plant(sys, initial_state=np.array([1.0, 2.0]))


class TestLoggedPlant(unittest.TestCase):
    """Unittest for the LoggedPlant class."""

    def test_logging_and_maxlen(self):
        """Test history logging and fixed-length deque behavior."""
        sys = LtiSystem(
            transition_matrix=np.array([[1.0]]),
            control_matrix=np.array([[1.0]]),
            output_matrix=np.array([[1.0]]))
        plant = LoggedPlant(sys,
                            initial_state=np.array([0.0]),
                            maxlen=3)

        # simulate 5 steps
        for _ in range(5):
            plant.step(np.array([1.0]))

        # maxlen is 3, so we expect only the last 3 steps
        np.testing.assert_allclose(plant.get_input_history(),
                                   np.array([[1.0], [1.0], [1.0]]))
        np.testing.assert_allclose(plant.get_state_history(),
                                   np.array([[3.0], [4.0], [5.0]]))
        np.testing.assert_allclose(plant.get_output_history(),
                                   np.array([[3.0], [4.0], [5.0]]))


if __name__ == '__main__':
    unittest.main()
