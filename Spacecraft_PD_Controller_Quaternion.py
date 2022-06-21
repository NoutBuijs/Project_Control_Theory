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
K_u = -0.0001/(np.pi/1000)
# K_p = np.array([K_u*0.9, K_u*0.7, K_u*0.8, 0])
# K_d = np.array([0.1*K_u*(330-148), 0.2*K_u*(302.7-141.1), 0.1*K_u*(116.7-103), 0])
K_p = np.array([K_u*10, K_u*10, K_u*0.8, 0])
K_d = np.array([0.8*K_u*2*(520-290), 0.8*K_u*2*(383-203), 0.1*K_u*(153.2-109), 0])

# kp1, kp2, kp3 = np.ones(3) * 1/10*0.8
# d = np.sqrt(2)/2
# kd2 = 2*d*np.sqrt(kp2*J[1,1])
# u = np.sqrt((kp1*kp3)/(J[0,0]*J[2,2]))
# v = J[2,2]*(kp1 + kp3)
# w = (-kp3 + u*J[2,2])/(kp1 - u*J[0,0])
# kd1 = np.sqrt((v - u*J[0,0]*J[2,2]*(4*d**2 - 2))/((J[0,0])/(4*d**2*J[0,0]) + (w)/(2*d**2) + (w**2*J[0,0])/(4*d**2*J[2,2])))
# kd3 = w*kd1
# K_p = np.array([kp1, kp2, kp3, 0])*-1
# K_d = np.array([kd1,kd2, kd3, 0])*-1

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
pd = util.Controller(K_p, K_d, sampling_frequency=1 / 0.1, response_function = "Quaternion_PD")
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

# np.savetxt("Quat_sim_999.csv", np.vstack((y.t, y.y[:4])), delimiter=",")