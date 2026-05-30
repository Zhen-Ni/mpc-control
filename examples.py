#!/usr/bin/env python3

import numpy as np

from discrete import LtiSystem
from mpc import Mpc

import matplotlib.pyplot as plt


A = np.array([0.9, 1., 0., 0.5]).reshape(2, 2)
B = np.array([0., 1.]).reshape(2, 1)
C = np.array([1., 0.]).reshape(1, 2)
s = LtiSystem(A, B, C)
x0 = np.array([0., 0.])
horizon = 100
target_output = np.zeros([horizon, s.n_output]) + 1.
output_weighting_matrix = np.eye(s.n_output)
control_weighting_matrix = np.eye(s.n_control)
controller = Mpc(s, horizon, x0, target_output,
                 output_weighting_matrix,
                 control_weighting_matrix)
predicted_control = controller.solve()
predicted_state = s.get_state(x0, predicted_control)
predicted_output = s.get_output(predicted_state)

u0 = np.zeros([1])
xi = s.get_state(x0, u0.reshape(1, -1)).reshape(-1)
y = []
u = []
for i in range(horizon):
    mpci = Mpc(s, horizon, xi, target_output,
               output_weighting_matrix, control_weighting_matrix)
    ui = mpci.solve()
    xi = s.get_state(xi, ui[0:1]).reshape(-1)
    y.append(s.get_output(xi.reshape(1, -1)).reshape(-1))
    u.append(ui[0].reshape(-1))


fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(211)
ax.plot(predicted_output, label='predicted @ step 0')
ax.plot(y, label='actual')
ax.legend()
ax.grid()
ax.set_ylabel('output')

ax2 = fig.add_subplot(212, sharex=ax)
ax2.plot(predicted_control, label='predicted @ step 0')
ax2.plot(u, label='actual')
ax2.legend()
ax2.grid()
ax2.set_ylabel('control')


plt.tight_layout(pad=0.1)

plt.show()
