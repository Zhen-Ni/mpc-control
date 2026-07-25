# A Python library for model predictive control.

A Python library for Model Predictive Control (MPC), integrating system modeling, state estimation, parameter identification, and Quadratic Programming (QP) based MPC solvers.

## Features

- **System Models (`mpc.discrete`)**: Supports discrete-time system modeling, including Linear Time-Invariant (LTI), Affine Time-Invariant (ATI), Nonlinear, and Homogeneous systems.
- **Model Predictive Control (`mpc.mpc`)**: Formulates and solves QP problems using the OSQP solver. Supports output, control, and control delta weighting, as well as constraints on output, control, and control rate.
- **State Estimation (`mpc.kalman`)**: Implements Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for state estimation of nonlinear systems.
- **Parameter Identification (`mpc.rls`)**: Provides Recursive Least Squares (RLS) algorithms for online system parameter identification.
