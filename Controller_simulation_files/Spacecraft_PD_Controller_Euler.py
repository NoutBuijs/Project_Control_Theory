import numpy as np
import Control_Classes as util
from matplotlib import pyplot as plt
import seaborn as sns
import quaternion as quat
mue = 3.986004418E14
R = (700E3 + 6371E3)
M_d = np.array([0.0001, 0.0001, 0.0001])
J = np.array([[124.531, 0,       0],
              [0,       124.586, 0],
              [0,       0,       0.704]])
n = np.sqrt(mue/R**3)

# kp1, kp2, kp3 = np.ones(3) * 0.0001/(np.pi/30)
# d = np.sqrt(2)/2
# kd2 = 2*d*np.sqrt(kp2*J[1,1])
# u = np.sqrt((kp1*kp3)/(J[0,0]*J[2,2]))
# v = J[2,2]*(kp1 + kp3)
# w = (-kp3 + u*J[2,2])/(kp1 - u*J[0,0])
# kd1 = np.sqrt((v - u*J[0,0]*J[2,2]*(4*d**2 - 2))/((J[0,0])/(4*d**2*J[0,0]) + (w)/(2*d**2) + (w**2*J[0,0])/(4*d**2*J[2,2])))
# kd3 = w*kd1
# K_p = np.array([kp1, kp2, kp3])*-1
# K_d = np.array([kd1,kd2, kd3])*-1
K_u = -0.0001/(np.pi/1000)
K_p = np.array([K_u*6, K_u*11, K_u*0.8])*0.1
K_d = np.array([0.1*K_u*2*(402-184), 0.2*K_u*2*(389-195), 0.2*K_u*(136.05-106.71)])
# K_u = 0.0001/(np.pi/1000)
# d = 3*n**2*(J[0,0]-J[2,2])
# se2 = (np.pi/1000)
# K_p2 = M_d[1]/se2 - d
# K_d2 = 2*np.sqrt(2)/2 * np.sqrt(M_d[1]/(se2*J[1,1]))
# K_p = np.array([0.6*K_u, 0.35*K_u, 0.8*K_u])*-1
# K_d = np.array([0.6*K_u*(197.599-196.370),0.01*K_u*(198.438-197.718), K_u*(50.05-21.36)])*-1

t = np.linspace(0, 1500, 15001)
x = np.array([1, 1, 1, 0, 0, 0])*np.radians(20)

u = np.zeros((np.size(t), 3))
cmd1 = [x > 99.9 and x <= 500 for x in t]
cmd2 = [x > 500 and x <= 900 for x in t]

theta = 70
u[cmd1] = np.radians(np.array([1, 1, 1])*theta)
u[cmd2] = np.radians(np.array([1, 1, 1])*-theta)

lim = np.size(t)
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1, response_function = "Euler_PD")
sim = util.simulator(pd, u[:lim], x, t[:lim], J,  n, M_d)
y = sim.run()


sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(1,3, figsize=(17,20))
fig.tight_layout()

name = "Euler Angles PD Controller"
fig.suptitle(f"{name} \n Kp: {K_p} \n Kd: {K_d}", fontsize=18)
q = quat.as_float_array(quat.from_euler_angles(y.y[:3].T, beta=None, gamma=None))
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
            axs[i].fill_between(y.t,
                                np.ones(np.size(y.t))*100,
                                np.ones(np.size(y.t))*100,
                                color="r", alpha=0.7, label="Singularity region")
            axs[i].legend(fontsize=16)

# fig1, axs1 = plt.subplots(2,4, figsize=(17,20))
# j = np.where(y.t < 1500)[0]
# for i in range(8):
#     if i < 3:
#         np.concatenate(axs1)[i].plot(y.t[j], q[j,i+1])
#     elif i == 3:
#         np.concatenate(axs1)[i].plot(y.t[j], q[j, 0])
#     elif i > 3 and i < 7:
#         np.concatenate(axs1)[i].plot(y.t, y.y[i-1])

# np.savetxt("Euler_sim_999.csv", np.vstack((y.t,y.y[:3])), delimiter=",")
