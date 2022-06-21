import numpy as np
from scipy import integrate as int

class simulator:

    def __init__(self, controller, u, x_0, t, J, n, M_d):

        self.controller = controller
        self.u = u
        self.t = t
        self.n = n
        self.J = J
        self.x_0 = x_0
        self.M_d = M_d
        self.M_c = 0.0
        self.samplings = -1

    def run(self):

        if np.shape(self.u)[1] == 3:
            x = int.solve_ivp(self.dxdt_euler, (self.t[0], self.t[-1]),
                              self.x_0, max_step=2)#, method="DOP853", rtol = 1E-13, atol=1E-13)
            self.reset()
            return x

        else:
            x = int.solve_ivp(self.dxdt_quat, (self.t[0], self.t[-1]), self.x_0)#, method="LSODA")#, method="DOP853", rtol = 1E-13, atol=1E-13)
            self.reset()
            return x

    def reset(self):
        self.samplings = -1
        self.M_c = 0.0

    def dxdt_quat(self, t, x):

        u = self.u[np.abs(self.t - t) == np.min(np.abs(self.t - t))][0]

        q = np.copy(x[:4])
        q[-1] = np.sign(q[-1]) * np.sqrt(1 - q[:3] @ q[:3])
        q_dot = np.copy(x[4:])
        q_dot[-1] = - np.sign(q_dot[-1]) * (1 / q[3]) * q_dot[:3] @ q[:3]

        Q = np.array([[0, -q[2], q[1]],
                      [q[2], 0, -q[0]],
                      [-q[1], q[0], 0]])

        L = 2 * np.array([[q[3],   q[2], -q[1], -q[0]],
                          [-q[2],  q[3],  q[0], -q[1]],
                          [ q[1], -q[0],  q[3], -q[2]],
                          [ q[0],  q[1],  q[2],  q[3]]])

        L_inv = 1/2 * np.array([[q[3], -q[2], q[1], q[0]],
                                [q[2], q[3], -q[0], q[1]],
                                [-q[1], q[0], q[3], q[2]],
                                [-q[0], -q[1], -q[2], q[3]]])

        C = (q[3]**2 - q[:3] @ q[:3])*np.identity(3) + 2 * q[:3].reshape((-1,1)) @ q[:3].reshape((1,-1)) - 2 * q[3] * Q

        # C = np.array([[1-2*(q[1]**2 + q[2]**2), 2*(q[0]* q[1] + q[2] * q[3]), 2*(q[0] * q[2] - q[1] * q[3])],
        #               [2*(q[1] * q[0] - q[2] * q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1] * q[2] + q[0] * q[3])],
        #               [2*(q[2] * q[0] + q[1] * q[3]), 2*(q[2] * q[1] - q[0] * q[3]), 1-2*(q[0]**2 + q[1]**2)]])

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        omega_b = L[:3, :3] @ q_dot[:3]

        Omega_b = np.array([[ 0,         -omega_b[2], omega_b[1]],
                            [ omega_b[2], 0,         -omega_b[0]],
                            [-omega_b[1], omega_b[0], 0         ]])
        print(q_dot[:3], -1/2*Omega_b@q[:3] + 1/2*q[3]*omega_b)
        # q_dot[:3] = -1/2*Omega_b@q[:3] + 1/2*q[3]*omega_b
        # q_dot[3] = -1/2*omega_b@q[:3]

        L_dot = 2 * np.array([[ q_dot[3],  q_dot[2], -q_dot[1], -q_dot[0]],
                              [-q_dot[2],  q_dot[3],  q_dot[0], -q_dot[1]],
                              [ q_dot[1], -q_dot[0],  q_dot[3], -q_dot[2]],
                              [ q_dot[0],  q_dot[1],  q_dot[2],  q_dot[3]]])


        # C_dot = -Omega_b @ C
        C_dot = np.array([[-4*(q[1]*q_dot[1] + q[2]*q_dot[2]), 2*(q[0]*q_dot[1] + q_dot[0]*q[1] + q_dot[2]*q_dot[3] + q_dot[2]*q[3]), 2*(q[0]*q_dot[2] + q_dot[0]*q[2] - q[1]*q_dot[3] - q_dot[1]*q[3])],
                          [2*(q[1]*q_dot[0] + q_dot[1]*q[0] - q[2]*q_dot[3] - q_dot[2]*q[3]), -4*(q[0]*q_dot[0] + q[2]*q_dot[2]), 2*(q[1]*q_dot[2] + q_dot[1]*q[2] + q[0]*q_dot[3] + q_dot[0]*q[3])],
                          [2*(q[2]*q_dot[0] + q_dot[2]*q[0] + q[1]*q_dot[3] + q_dot[1]*q[3]), 2*(q[2]*q_dot[1] + q_dot[2] *q[1] - q[0] * q_dot[3] - q_dot[0] * q[3]), -4*(q[0]*q_dot[0] + q[1]*q_dot[1])]])
        #print(C_dot, "\n", C_dot_1, "\n", C, "\n", C_1, "\n")

        # print(q_dot, "\n", q_dot_1, q_dot_4)
        R_dot = np.array([C_dot[0,1], C_dot[1,1], C_dot[2,1]])

        omega = omega_b - self.n * R

        Omega = np.array([[ 0,       -omega[2], omega[1]],
                          [omega[2], 0,        -omega[0]],
                          [-omega[1], omega[0], 0       ]])

        if 1/self.controller.f == 0 or t * self.controller.f > self.samplings:
            M_c = self.controller.response_q(u, q, omega_b)
            self.samplings += 1

        x_dot = np.zeros(8)
        x_dot[:4] = q_dot
        x_dot[4:-1] = L_inv[:3,:3] @ np.diag(1/np.diag(self.J)) \
                                @ (-self.J @ (L_dot[:3,:3] @ q_dot[:3] - self.n * R_dot) \
                                   - Omega @ self.J @ omega \
                                   + 3 * self.n**2 * G @ self.J @ g + self.M_d + M_c)
        # x_dot[-1] = - (q[:3] @ x_dot[4:-1] + x_dot[:4] @ x_dot[:4]) / q[3]
        print(t)
        return x_dot

    def dxdt_euler(self, t, x):

        u = self.u[np.abs(self.t - t) == np.min(np.abs(self.t - t))][0]

        theta = x[:3]
        theta_dot = x[3:]

        C3 = np.array([[np.cos(theta[2]), np.sin(theta[2]), 0],
                       [-np.sin(theta[2]), np.cos(theta[2]), 0],
                       [0, 0, 1]])

        C2 = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
                       [0, 1, 0, ],
                       [np.sin(theta[1]), 0, np.cos(theta[1])]])

        C1 = np.array([[1, 0, 0, ],
                       [0, np.cos(theta[0]), np.sin(theta[0])],
                       [0, -np.sin(theta[0]), np.cos(theta[0])]])

        L = np.array([[1, 0, -np.sin(theta[1])],
                      [0, np.cos(theta[0]), np.sin(theta[0]) * np.cos(theta[1])],
                      [0, -np.sin(theta[0]), np.cos(theta[0]) * np.cos(theta[1])]])

        L_inv = 1 / np.cos(theta[1]) * np.array([[np.cos(theta[1]), np.sin(theta[0]) * np.sin(theta[1]), np.cos(theta[0]) * np.sin(theta[1])],
                                             [0, np.cos(theta[0]) * np.cos(theta[1]), -np.sin(theta[0]) * np.cos(theta[1])],
                                             [0, np.sin(theta[0]), np.cos(theta[0])]])

        C = C1 @ C2 @ C3

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        omega_b = L @ theta_dot[:3]

        L_dot = np.array([[0, 0, -np.cos(theta[1]) * theta_dot[1]],
                              [0, -np.sin(theta[0]) * theta_dot[0], np.cos(theta[0]) * np.cos(theta[1]) * theta_dot[0] - np.sin(theta[0]) * np.sin(theta[1]) * theta_dot[1]],
                              [0, -np.cos(theta[0]) * theta_dot[0], -np.sin(theta[0]) * np.cos(theta[1]) * theta_dot[0] - np.cos(theta[0]) * np.sin(theta[1]) * theta_dot[1]]])

        Omega_b = np.array([[ 0,         -omega_b[2], omega_b[1]],
                            [ omega_b[2], 0,         -omega_b[0]],
                            [-omega_b[1], omega_b[0], 0         ]])

        C_dot = -Omega_b @ C

        R_dot = np.array([C_dot[0,1], C_dot[1,1], C_dot[2,1]])

        omega = omega_b - self.n * R

        Omega = np.array([[ 0,       -omega[2], omega[1]],
                          [omega[2], 0,        -omega[0]],
                          [-omega[1], omega[0], 0       ]])

        if 1/self.controller.f == 0 or t * self.controller.f >= self.samplings:
            M_c = self.controller.response_euler(u, theta, omega_b)
            self.M_c = M_c
            self.samplings += 1
        else:
            M_c = self.M_c

        x_dot = np.zeros(6)
        x_dot[:3] = theta_dot
        x_dot[3:] = L_inv @ np.diag(1/np.diag(self.J)) \
                                @ (-self.J @ (L_dot @ theta_dot - self.n * R_dot) \
                                   - Omega @ self.J @ omega \
                                   + 3 * self.n**2 * G @ self.J @ g + self.M_d + M_c)
        print(t)
        return x_dot

class PD_controller:

    def __init__(self, K_p, K_d, sampling_frequency = np.inf):
        self.K_p = K_p
        self.K_d = K_d
        self.e = np.zeros(3, dtype=object)
        self.f = sampling_frequency

    def response_q(self, u, x, omega_b):

        L_inv = np.array([[ x[3], -x[2],  x[1], x[0]],
                          [ x[2],  x[3], -x[0], x[1]],
                          [-x[1],  x[0],  x[3], x[2]],
                          [-x[0], -x[1], -x[2], x[3]]])

        L_c = np.array([[ u[3],  u[2], -u[1], -u[0]],
                        [-u[2],  u[3],  u[0], -u[1]],
                        [ u[1], -u[0],  u[3], -u[2]],
                        [ u[0],  u[1],  u[2],  u[3]]])

        e_dot = np.zeros(4)
        e_dot[:3] = 1/2 * L_inv[:3,:3] @ omega_b
        e_dot[3] = L_inv[3,:3] @ omega_b

        return (self.K_p * L_c @ x + self.K_d * e_dot)[:3]

    def response_euler(self, u, x, omega_b):

        L_inv = 1/np.cos(x[1]) * np.array([[np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.sin(x[1])],
                                           [0, np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.cos(x[1])],
                                           [0, np.sin(x[0]), np.cos(x[0])]])

        e_dot = 1/2 * L_inv[:3,:3] @ omega_b
        return self.K_p * (x-u) + self.K_d * e_dot
