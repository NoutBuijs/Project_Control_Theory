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
K_u = 0.0001/(np.pi/100000)
d = 3*n**2*(J[0,0]-J[2,2])
se2 = (np.pi/100000)
K_p2 = M_d[1]/se2 - 2*d
K_d2 = 4*np.sqrt(2)/2 * np.sqrt(M_d[1]/(se2*J[1,1]*2))
# K_p = np.array([K_u*0.9, K_u*0.7, K_u*0.8, 0])*-1
# K_d = np.array([0.1*K_u*(330-148), 0.2*K_u*(302.7-141.1), 0.1*K_u*(116.7-103), 0])-1
K_p = np.array([K_u*10, K_u*10, K_u*0.8, 0])*-1
K_d = np.array([0.8*K_u*2*(520-290), 0.8*K_u*2*(383-203), 0.1*K_u*(153.2-109), 0])*-1
# K_p = np.array([0.8*K_u, K_p2, 0.8*K_u, 0])*-1
# K_d = np.array([0.1*K_u*(168.6-113.8), 56*K_d2, 0.1*K_u*(372.4 - 316.2), 0])*-1
x = np.array([1/np.sqrt(3)*np.sin(np.radians(20)/2),
              1/np.sqrt(3)*np.sin(np.radians(20)/2),
              1/np.sqrt(3)*np.sin(np.radians(20)/2),
              np.cos(np.radians(20)/2),
              0, 0, 0])

t = np.linspace(0,1500, 1501)


u = np.zeros((np.size(t), 4))
theta = 70
u[:,-1] = 1
cmd1 = [x > 99.9 and x <= 500 for x in t]
cmd2 = [x > 500 and x <= 900 for x in t]


u[cmd1, :3] = np.array([1, 1, 1])*1/np.sqrt(3)*np.sin(np.radians(theta)/2)
u[cmd1, -1] = np.cos(np.radians(theta)/2)
u[cmd2, :3] = np.array([1, 1, 1])*1/np.sqrt(3)*np.sin(np.radians(-theta)/2)
u[cmd2, -1] = np.cos(np.radians(-theta)/2)

lim = 1500
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1, response_function = "Quaternion_PD")
sim = util.simulator(pd, u[:lim], x, t[:lim], J,  n, M_d)
y = sim.run()
y.y[3] = np.sign(y.y[3])  * np.sqrt(1 - np.linalg.norm(y.y[:3], axis = 0)**2)
sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(1,4, figsize=(17,20))
fig.tight_layout()

# for i, solution in enumerate(y.y):
#         np.concatenate(axs)[i].plot(y.t, solution, c = "k", ls = "--")
#         if i < 4:
#             np.concatenate(axs)[i].plot(t[:lim], u[:lim, i], c = "r")
name = "Quaternion PD Controller"
fig.suptitle(f"{name} Kp: {K_p[:3]} \n Kd: {K_d[:3]}",fontsize=18)
for i, solution in enumerate(y.y[:4]):
        axs[i].plot(y.t, solution, c = "k", ls = "--", label="Controller")
        axs[i].set_xlim(np.min(y.t), np.max(y.t))
        axs[i].set_ylim(np.min((np.min(solution)*0.9,np.min(solution)*1.1)) , np.max(solution)*1.1)
        axs[i].set_ylabel(u"x{0:.0f} [-]".format(i), fontsize=18)
        axs[i].set_xlabel(u"t [s]", fontsize=18)
        axs[i].set_yticklabels(labels=np.round(axs[i].get_yticks(), 2), fontsize=14)
        axs[i].set_xticklabels(labels=np.round(axs[i].get_xticks()).astype(int), fontsize=12)
        if i < 4:
            axs[i].plot(t[:lim], u[:lim, i], c = "r", label="Command")
        if i == 3:
            # axs[i].fill_between(y.t,
            #                     np.ones(np.size(y.t)) * 0,
            #                     np.ones(np.size(y.t)) * 0.1,
            #                     color="r", alpha=0.7, label="Singularity region")
            axs[i].legend(fontsize=16)
# np.savetxt("Quat_sim_999.csv", np.vstack((y.t, y.y[:4])), delimiter=",")