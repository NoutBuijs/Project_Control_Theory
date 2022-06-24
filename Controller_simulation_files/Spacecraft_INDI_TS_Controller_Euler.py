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
w = 0.3
w1 = 1
K_p = np.array([w, w, w])*-0.35
K_d = np.array([w1, w1, w1])*-0.006

x = np.array([1, 1, 1, 0, 0, 0])*np.radians(20)

t = np.linspace(0,1500, 1500)


u = np.zeros((np.size(t), 3))
cmd1 = [x > 99.9 and x <= 500 for x in t]
cmd2 = [x > 500 and x <= 900 for x in t]

theta = 40
u[cmd1, :3] = np.array([1, 1, 1])*np.radians(theta)
u[cmd2, :3] = np.array([1, 1, 1])*np.radians(-theta)

lim = np.size(t)
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1,
                     response_function = "Euler_INDI_TS", J = J, n = n)
sim = util.simulator(pd, u[:lim], x, t[:lim], J,  n, M_d)
y = sim.run()
sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(1,3, figsize=(17,20))
name = "Euler Angles INDI-TS Controller"
fig.tight_layout()
fig.suptitle(f"{name} \n Kp: {K_p} \n Kd: {K_d}", fontsize=18)
for i, solution in enumerate(y.y[:3]):
        axs[i].plot(y.t, solution, c = "k", ls = "--", label="Controller")
        axs[i].set_xlim(np.min(y.t), np.max(y.t))
        axs[i].set_ylim(np.min((np.min(solution)*0.9,np.min(solution)*1.1)), np.max(solution)*1.1)
        axs[i].set_ylabel(u"x{0:.0f} [rad]".format(i), fontsize=16)
        axs[i].set_xlabel(u"t [s]", fontsize=16)
        axs[i].set_yticklabels(labels=np.round(axs[i].get_yticks(), 2), fontsize=14)
        axs[i].set_xticklabels(labels=np.round(axs[i].get_xticks()).astype(int), fontsize=12)
        if i < 3:
            axs[i].plot(t[:lim], u[:lim, i], c = "r", label="Command")

        if i == 1:
            axs[i].fill_between(y.t,
                                               np.ones(np.size(y.t))*np.pi/2*1.1,
                                               np.ones(np.size(y.t))*np.pi/2*0.9,
                                               color = "r", alpha=0.7, label="Singularity region")
        if i == 2:
            # axs[i].fill_between(y.t,
            #                     np.ones(np.size(y.t))*100,
            #                     np.ones(np.size(y.t))*100,
            #                     color="r", alpha=0.7, label="Singularity region")
            axs[i].legend(fontsize=16)

# np.savetxt("Quat_NDI_sim_999.csv", np.vstack((y.t, y.y[:4])), delimiter=",")