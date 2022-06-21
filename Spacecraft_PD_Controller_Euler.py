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
K_p = np.array([K_u*6, K_u*6, K_u*0.8])*0.1
K_d = np.array([0.1*K_u*2*(402-184), 0.2*K_u*2*(389-195), 0.2*K_u*(136.05-106.71)])


t = np.linspace(0, 1500, 1501)
x = np.array([0, 0, 0, 0, 0, 0])

u = np.zeros((np.size(t), 3))
cmd1 = [x > 99.9 and x <= 500 for x in t]
cmd2 = [x > 500 and x <= 900 for x in t]

theta = 10
u[cmd1] = np.radians(np.array([1, 1, 1])*theta)
u[cmd2] = np.radians(np.array([1, 1, 1])*-theta)

lim = 1500
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1, response_function = "Euler_PD")
sim = util.simulator(pd, u[:lim], x, t[:lim], J,  n, M_d)
y = sim.run()


sns.set_theme()
sns.set_palette("rocket")
fig, axs = plt.subplots(2,3, figsize=(17,20))
fig.suptitle(f"K_p = {K_p} \n K_d = {K_d}")
q = quat.as_float_array(quat.from_euler_angles(y.y[:3].T, beta=None, gamma=None))
for i, solution in enumerate(y.y):
        np.concatenate(axs)[i].plot(y.t, solution, c = "k", ls = "--")
        if i < 3:
            np.concatenate(axs)[i].plot(t[:lim], u[:lim, i], c = "r")

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
