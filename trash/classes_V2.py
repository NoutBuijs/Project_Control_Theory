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
        self.samplings = -1

    def run(self):

        if np.size(self.u[0]) == 3:
            raise NotImplementedError

        else:
            x = int.solve_ivp(self.dxdt_quat, (self.t[0], self.t[-1]), self.x_0)
            self.reset()
            return x

    def reset(self):
        self.samplings = -1

    def dxdt_quat(self, t, x):

        u = self.u[np.abs(self.t - t) == np.min(np.abs(self.t - t))][0]

        q = np.zeros(4)
        q[:3] = x[:3]
        q[-1] = np.sqrt(1 - np.linalg.norm(q[:3])**2)

        A = np.array([[0, -q[2], q[1]],
                      [q[2], 0, -q[0]],
                      [-q[1], q[0], 0]])

        C = (q[3]**2 - q[:3] @ q[:3]) * np.identity(3) + 2 * q[:3]**2 - 2*q[:3] @ A

        c = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        Q = np.array([[ q[3],  q[2], -q[1]],
                      [-q[2],  q[3],  q[0]],
                      [ q[1], -q[0],  q[3]]])

        Q_inv = np.array([[q[3], -q[2], q[1]],
                          [q[2], q[3], -q[0]],
                          [-q[1], q[0], q[3]]])

        q_dot = np.zeros(4)
        q_dot[:3] = x[3:]
        omega_b = 2 * Q @ q_dot[:3]
        q_dot[-1] = -1/2 * omega_b @ q[:3]

        Q_dot = np.array([[ q_dot[3],  q_dot[2], -q_dot[1]],
                          [-q_dot[2],  q_dot[3],  q_dot[0]],
                          [ q_dot[1], -q_dot[0],  q_dot[3]]])

        k = np.array([C[0,1], C[1,1], C[2,1]])

        k_dot = np.array([ 2*(q_dot[0]*q[1] + q[0]*q_dot[1] + q_dot[2]*q[3] + q[2]*q_dot[3]),
                          -4*(q[0] + q[2]),
                           2*(q_dot[2]*q[1] + q[2]*q_dot[1] - q_dot[0]*q[3] - q[0]*q_dot[3])])

        omega = omega_b - self.n * k

        Omega = np.array([[ 0,       -omega[2], omega[1]],
                          [ omega[2], 0,       -omega[0]],
                          [-omega[1], omega[0], 0       ]])

        if 1/self.controller.f == 0 or t * self.controller.f > self.samplings:
            M_c = self.controller.response_q(u, q, omega_b)
            self.samplings += 1

        x_dot = np.zeros(6)
        x_dot[:3] = q_dot[:3]
        x_dot[3:] = Q_inv @ np.diag(np.diag(1/self.J)) \
                      @ (1/2 * (-self.J @ (2 * Q_dot @ q_dot[:3] - self.n * k_dot) \
                                - Omega @ self.J @ omega \
                                + 3 * self.n**2 * G @ self.J @ c + self.M_d + M_c))
        return x_dot


class PD_controller:

    def __init__(self, A, K_p, K_d, sampling_frequency = np.inf):

        self.A = A
        self.K_p = K_p
        self.K_d = K_d
        self.f = sampling_frequency

    def response_q(self, u, x, omega_b):

        Q_inv = np.array([[ x[3], -x[2],  x[1]],
                          [ x[2],  x[3], -x[0]],
                          [-x[1],  x[0],  x[3]]])

        Q_c = np.array([[ u[3],  u[2], -u[1]],
                        [-u[2],  u[3],  u[0]],
                        [ u[1], -u[0],  u[3]]])

        e_dot = 1/2 * Q_inv @ omega_b
        return self.K_p * Q_c @ x[:3] + self.K_d * e_dot

    def reset(self, e_start = np.NaN):

        if np.isfinite(e_start):
            self.e[-1] = e_start
        else:
            self.e[-1] = self.e_start
        return self.e[-1]

