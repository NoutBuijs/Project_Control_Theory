import numpy as np
import classes_V3 as util
from matplotlib import pyplot as plt
import seaborn as sns

mue = 3.986004418E14
R = (700E3 + 6371E3)
n = np.sqrt(mue/R**3)
M_d = np.array([0.0001, 0.0001, 0.0001])
K_p = np.array([1, 1, 1, 0])*0.0
K_d = np.array([1, 1, 1, 0])*0.0

x = np.array([0, 0, 0, 1,
              0, 0, 0, 0])

J = np.array([[124.531, 0,       0],
              [0,       124.586, 0],
              [0,       0,       0.704]])

t = np.linspace(0,930.779, 1501)

theta = 0
u = np.zeros((np.size(t), 4))
u_ref = np.zeros(4)
u_ref[:3] = 1/np.sqrt(3)*np.sin(theta/2)
u_ref[-1] = np.cos(theta/2)
u[:] = u_ref

pd = util.PD_controller(K_p, K_d)
sim = util.simulator(pd, u, x, t, J,  n, M_d)
y = sim.run()
y.y[3] = np.sign(y.y[3])  * np.sqrt(1 - np.linalg.norm(y.y[:3], axis = 0)**2)
y.y[7] = - (1/y.y[3]) * np.sum(y.y[:3] * y.y[4:-1], axis=0)
sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(2,4, figsize=(17,20))
for i, solution in enumerate(y.y):
        np.concatenate(axs)[i].plot(y.t, solution)

fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(17,20))
ax1.plot(y.t, np.arccos(y.y[3])*2)
ax2.plot(y.t, np.linalg.norm(y.y[:4], axis = 0))
# y = np.zeros((8, np.size(t)))
# dt = t[1] - t[0]
# y[:4,0] = np.array([0,0,0,1])
# for i,time in enumerate(t[:-1]):
#     y[:,i+1] = y[:,i] + sim.dxdt_quat(t[i], y[:,i])*dt
#
# sns.set_theme()
# sns.set_palette("rocket")
# fig, axs = plt.subplots(2,4, figsize=(17,20))
# for i, solution in enumerate(y):
#         np.concatenate(axs)[i].plot(t, solution)