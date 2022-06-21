import numpy as np
import classes_V2 as util
from matplotlib import pyplot as plt
import seaborn as sns

mue = 3.986004418E14
R = (700E3 + 6371E3)
K_p = np.array([1, 1, 1])*0.0
K_d = np.array([1, 1, 1])*0.0

x = np.array([0, 0, 0,
              0, 0, 0])

J = np.array([[124.531, 0,       0],
              [0,       124.586, 0],
              [0,       0,       0.704]])

t = np.linspace(0, 170, 1501)

theta = 0
u = np.zeros((np.size(t), 4))
u_ref = np.zeros(4) * 1/np.sqrt(3)*np.sin(theta/2)
u_ref[-1] = np.cos(theta/2)

u[:] = u_ref
n = np.sqrt(mue/R**3)
M_d = np.array([0.0001, 0.0001, 0.0001])

pd = util.PD_controller(K_p, K_d)
sim = util.simulator(pd, u, x, t, J, n, M_d)
y = sim.run()


sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(2,2, figsize=(17,20))
for i, solution in enumerate(y.y[:4]):

    if i < 3:
        np.concatenate(axs)[i].plot(y.t, solution)
    else:
        np.concatenate(axs)[i].plot(y.t, np.sqrt(1 - np.linalg.norm(y.y[:3], axis=0)**2))

# Note on to do: conservation of q4 is flawed; it cant handle negative q4
