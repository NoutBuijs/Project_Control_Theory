import numpy as np
import Control_Classes as util
from matplotlib import pyplot as plt
import seaborn as sns

mue = 3.986004418E14
R = (700E3 + 6371E3)
n = np.sqrt(mue/R**3)
J = np.array([[124.531, 0,       0],
              [0,       124.586, 0],
              [0,       0,       0.704]])
M_d = np.array([0.0001, 0.0001, 0.0001])
# w = 0.05
# damp = np.sqrt(2)/3
# K_p = np.array([w**2, w**2, (w*1.5)**2, 0])
# K_d = np.array([2*w*damp, 2*w*damp, 2.5*w*damp, 0])
w = 1
damp = np.sqrt(2)/2
K_p = np.array([w**2, w**2, w**2, 0])
K_d = np.array([2*w*damp, 2*w*damp, 2*w*damp, 0])

x = np.array([0, 0, 0, 1,
              0, 0, 0])

t = np.linspace(0,1500, 1501)


u = np.zeros((np.size(t), 4))
u[:,-1] = 1
cmd1 = [x > 99.9 and x <= 500 for x in t]
cmd2 = [x > 500 and x <= 900 for x in t]

theta = 70
u[cmd1, :3] = np.array([1, 1, 1])*1/np.sqrt(3)*np.sin(np.radians(theta)/2)
u[cmd1, -1] = np.cos(np.radians(theta)/2)
u[cmd2, :3] = np.array([1, 1, 1])*1/np.sqrt(3)*np.sin(np.radians(-theta)/2)
u[cmd2, -1] = np.cos(np.radians(-theta)/2)

lim = 1500
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1,
                     response_function = "Quaternion_NDI", J = J, n = n)
sim = util.simulator(pd, u[:lim], x, t[:lim], J,  n, M_d)
y = sim.run()
y.y[3] = np.sign(y.y[3])  * np.sqrt(1 - np.linalg.norm(y.y[:3], axis = 0)**2)
sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(2,4, figsize=(17,20))
fig.suptitle(f"K_p = {K_p} \n K_d = {K_d}")
for i, solution in enumerate(y.y):
        np.concatenate(axs)[i].plot(y.t, solution, c = "k", ls = "--")
        if i < 4:
            np.concatenate(axs)[i].plot(t[:lim], u[:lim, i], c = "r")

# np.savetxt("Quat_NDI_sim_999.csv", np.vstack((y.t, y.y[:4])), delimiter=",")