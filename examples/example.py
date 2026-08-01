import numpy as np
import mpc

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
