[![PyPI version](https://badge.fury.io/py/mpc-control.svg)](https://badge.fury.io/py/mpc-control)
[![CI Status](https://github.com/Zhen-Ni/mpc-control/actions/workflows/ci.yml/badge.svg)](https://github.com/Zhen-Ni/mpc-control/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

# A Python library for model predictive control.

A Python library for Model Predictive Control (MPC), integrating system modeling, state estimation, parameter identification, and Quadratic Programming (QP) based MPC solvers.

## Requirements

- Python 3.9+ (Tested with Python 3.12)
- NumPy
- SciPy
- OSQP

## Model predictive control

### System Prediction Model

For discrete time-invariant systems:

$$
\begin{aligned}
x[n+1] &= A x[n] + B u[n] + w \\
y[n]   &= C x[n] + v
\end{aligned}
$$

Prediction over horizon
$N$:

$$
\begin{aligned}
X &= [x_1^T, x_2^T, \dots, x_N^T]^T \\
U &= [u_0^T, u_1^T, \dots, u_{N-1}^T]^T \\
Y &= [y_1^T, y_2^T, \dots, y_N^T]^T
\end{aligned}
$$

The predicted state and output sequences can be expressed as:

$$
\begin{aligned}
X &= M_x x_0 + M_u U + M_w w \\
Y &= \bar{C} X + V = \bar{C} M_x x_0 + \bar{C} M_u U + \bar{C} M_w w + V
\end{aligned}
$$

where $M_x$, $M_u$, and $M_w$ are block matrices defined as:

$$
M_x = \begin{bmatrix} A \\ A^2 \\ \cdots \\ A^N \end{bmatrix}
$$

$$
M_u = \begin{bmatrix} 
B & 0 & \dots & 0 \\ 
AB & B & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
A^{N-1}B & A^{N-2}B & \dots & B 
\end{bmatrix}
$$

$$
M_w = \begin{bmatrix} 
I & 0 & \dots & 0 \\ 
A+I & I & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
\sum_{i=0}^{N-1} A^i & \sum_{i=0}^{N-2} A^i & \dots & I 
\end{bmatrix}
$$

and $\bar{C}$ and $V$ are defined as:

$$
\bar{C} = \text{diag}(C, C, \dots, C), \quad
V = [v^T, v^T, \dots, v^T]^T
$$

### Cost Function

The optimization objective is to minimize the cost function:

$$
J = (Y - Y_{ref})^T \bar{Q} (Y - Y_{ref}) + U^T \bar{R} U + \Delta U^T \bar{R}_{\Delta} \Delta U
$$

where $\bar{Q}$, $\bar{R}$, and $\bar{R}_{\Delta}$ are block-diagonal weighting matrices for output, control, and control delta respectively. The control delta is defined as:

$$
\Delta U = \bar{D} U - U_{last}
$$

where $U_{last} = [u_{-1}^T, 0, \dots, 0]^T$ ($u_{-1}$ is the previous control input), and $\bar{D}$ is the control delta matrix defined as:

$$
\bar{D} = \begin{bmatrix} 
I & 0 & \dots & 0 \\ 
-I & I & \dots & 0 \\ 
0 & -I & \dots & 0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
0 & 0 & \dots & I 
\end{bmatrix}
$$

### QP Formulation

Let $E_y = \bar{C} M_x x_0 + \bar{C} M_w w + V - Y_{ref}$. Expanding the cost function and ignoring constant terms, we obtain:

$$
J = \frac{1}{2} U^T (2 M_u^T \bar{C}^T \bar{Q} \bar{C} M_u + 2 \bar{R} + 2 \bar{D}^T \bar{R}_{\Delta} \bar{D}) U + (2 M_u^T \bar{C}^T \bar{Q} E_y - 2 \bar{D}^T \bar{R}_{\Delta} U_{last})^T U
$$

This can be mapped to the standard OSQP form ($\min \frac{1}{2} U^T P U + q^T U$). The actual $P$ and $q$ computed in the code are (without the factor of 2):

$$
\begin{aligned}
P &= M_u^T \bar{C}^T \bar{Q} \bar{C} M_u + \bar{R} + \bar{D}^T \bar{R}_{\Delta} \bar{D} \\
q &= M_u^T \bar{C}^T \bar{Q} E_y - \bar{D}^T \bar{R}_{\Delta} U_{last}
\end{aligned}
$$

### Constraints

The problem is subject to the following constraints:
- **Output constraints**: $l_{y} \leq \bar{C} M_u U + \bar{C} M_x x_0 + \bar{C} M_w w + V \leq u_{y}$
- **Control constraints**: $l_{u} \leq U \leq u_{u}$
- **Control rate constraints**: $l_{\Delta u} \leq \bar{D} U - U_{last} \leq u_{\Delta u}$ 

These linear constraints are compiled into the standard form $l \leq A_c U \leq u$ for the OSQP solver.

### Nonlinear Systems

For nonlinear systems, the controller linearizes the system dynamics along a given reference trajectory. At each time step $i$ within the prediction horizon, the system is linearized around the reference state $x_{ref, i}$ and control $u_{ref, i}$ to obtain a linear time-varying (LTV) model:

$$
\begin{aligned}
x[i+1] &\approx A_i x[i] + B_i u[i] + w_i \\
y[i]   &\approx C_i x[i] + v_i
\end{aligned}
$$

The QP problem is then formulated using these LTV matrices. The prediction matrices $M_x, M_u, M_w$ and the output mapping $\bar{C}$ become time-varying and are constructed iteratively over the horizon to reflect the changing linearization points. The reference trajectory for linearization can be provided to the solver via the `state_ref` and `control_ref` arguments.

## Features

- **System Models (`mpc.discrete`)**: Supports discrete-time system modeling, including Linear Time-Invariant (LTI), Affine Time-Invariant (ATI), Nonlinear, and Homogeneous systems.
- **Model Predictive Control (`mpc.mpc`)**: Formulates and solves QP problems using the OSQP solver. Supports output, control, and control delta weighting, as well as constraints on output, control, and control rate.
- **State Estimation (`mpc.kalman`)**: Implements Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for state estimation of nonlinear systems.
- **Parameter Identification (`mpc.rls`)**: Provides Recursive Least Squares (RLS) algorithms for online system parameter identification.

## Usage

Here is a basic example of how to define a system and solve an MPC problem:

```python
import numpy as np
import mpc_control as mpc

# 1. Define a discrete LTI system
# x[n+1] = A x[n] + B u[n]
# y[n]   = C x[n]
system = mpc.LtiSystem(
    transition_matrix=np.array([[1.0, 1.0],
                                [0.0, 1.0]]),
    control_matrix=np.array([[0.0],
                             [1.0]]),
    output_matrix=np.array([[1.0, 0.0]])
)

# 2. Initialize the MPC controller
horizon = 10
n_output = system.n_output
n_control = system.n_control

Q = np.stack([np.eye(n_output)] * horizon)  # Output weighting
R = np.stack([np.eye(n_control) * 0.1] * horizon)  # Control weighting

controller = mpc.Mpc(
    system=system,
    horizon=horizon,
    output_weighting=Q,
    control_weighting=R
)

# 3. Set up the problem and solve
target_output = np.zeros([horizon, n_output])
initial_state = np.array([1.0, 0.0])

u_optimal = controller.solve(
    target_output=target_output,
    initial_state=initial_state
)

print("Optimal control sequence:\n", u_optimal)
```
