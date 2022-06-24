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
        self.M_c_0 = 0.0
        self.M_c = self.M_c_0
        self.samplings_0 = 0
        self.samplings = self.samplings_0
        self.x_dot_0 = np.zeros(np.size(self.M_d))
        self.x_dot = self.x_dot_0
        self.t_0 = self.t[0]
        self.t_sens = self.t_0

    def run(self):

        if np.shape(self.u)[1] == 3:
            x = int.solve_ivp(self.dxdt_euler, (self.t[0], self.t[-1]), self.x_0, max_step = 0.1)#, rtol = 1E-5, atol=1E-5)#, method="DOP853")
            self.controller.reset()
            self.reset()
            return x

        else:
            x = int.solve_ivp(self.dxdt_quat, (self.t[0], self.t[-1]), self.x_0, max_step = 0.1)#,rtol = 1E-16, atol=1E-16, method="DOP853")
            self.controller.reset()
            self.reset()
            return x

    def reset(self):
        self.samplings = self.samplings_0
        self.x_dot = self.x_dot_0
        self.M_c = self.M_c_0
        self.t_sens = self.t_0

    def dxdt_quat(self, t, x):

        u = self.u[np.abs(self.t - t) == np.min(np.abs(self.t - t))][0]

        q = np.copy(x[:4])
        q[-1] = np.sign(q[-1]) * np.sqrt(1 - q[:3] @ q[:3])
        omega_b = np.copy(x[4:])

        Q = np.array([[0, -q[2], q[1]],
                      [q[2], 0, -q[0]],
                      [-q[1], q[0], 0]])

        C = (q[3]**2 - q[:3] @ q[:3])*np.identity(3) + 2 * q[:3].reshape((-1,1)) @ q[:3].reshape((1,-1)) - 2 * q[3] * Q

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        Omega_b = np.array([[ 0,         -omega_b[2], omega_b[1], -omega_b[0]],
                            [ omega_b[2], 0,         -omega_b[0], -omega_b[1]],
                            [-omega_b[1], omega_b[0], 0         , -omega_b[2]],
                            [ omega_b[0], omega_b[1], omega_b[2],  0         ]])

        C_dot = - Omega_b[:3, :3] @ C

        R_dot = np.array([C_dot[0,1], C_dot[1,1], C_dot[2,1]])

        omega = omega_b - self.n * R

        Omega = np.array([[ 0,       -omega[2], omega[1]],
                          [omega[2], 0,        -omega[0]],
                          [-omega[1], omega[0], 0       ]])

        if 1/self.controller.f == 0 or t * self.controller.f >= self.samplings:
            M_c = self.controller.response(u, q, omega_b, self.x_dot)
            M_c = np.vstack((self.controller.M_max, M_c))
            M_c = M_c[np.argmin(np.abs(M_c), axis=0), np.array([0,1,2])]
            self.M_c = M_c
            self.samplings += 1
        else:
            M_c = self.M_c

        x_dot = np.zeros(7)
        x_dot[:4] = - 1/2 * Omega_b @ q
        x_dot[4:] = np.diag(1/np.diag(self.J)) \
                                @ (- Omega @ self.J @ omega \
                                + 3 * self.n**2 * G @ self.J @ g \
                                + self.M_d + M_c) + self.n*R_dot
        print(t)
        if self.t_sens < t:
            self.t_sens = t
            self.x_dot = x_dot[4:]
        return x_dot

    def dxdt_euler(self, t, x):

        u = self.u[np.abs(self.t - t) == np.min(np.abs(self.t - t))][0]

        theta = x[:3]
        omega_b = x[3:]

        C3 = np.array([[np.cos(theta[2]), np.sin(theta[2]), 0],
                       [-np.sin(theta[2]), np.cos(theta[2]), 0],
                       [0, 0, 1]])

        C2 = np.array([[np.cos(theta[1]), 0, -np.sin(theta[1])],
                       [0, 1, 0, ],
                       [np.sin(theta[1]), 0, np.cos(theta[1])]])

        C1 = np.array([[1, 0, 0, ],
                       [0, np.cos(theta[0]), np.sin(theta[0])],
                       [0, -np.sin(theta[0]), np.cos(theta[0])]])

        L_inv = 1 / np.cos(theta[1]) * np.array([[np.cos(theta[1]), np.sin(theta[0]) * np.sin(theta[1]), np.cos(theta[0]) * np.sin(theta[1])],
                                             [0, np.cos(theta[0]) * np.cos(theta[1]), -np.sin(theta[0]) * np.cos(theta[1])],
                                             [0, np.sin(theta[0]), np.cos(theta[0])]])

        C = C1 @ C2 @ C3

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

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
            omega_b_dot = np.diag(1/np.diag(self.J)) \
                                @ (- Omega @ self.J @ omega \
                                + 3 * self.n**2 * G @ self.J @ g \
                                + self.M_d + self.M_c) + self.n*R_dot
            M_c = self.controller.response(u, theta, omega_b, omega_b_dot)
            M_c = np.vstack((self.controller.M_max, M_c))
            M_c = M_c[np.argmin(np.abs(M_c), axis=0), np.array([0,1,2])]
            self.M_c = M_c
            self.samplings += 1
        else:
            M_c = self.M_c

        x_dot = np.zeros(6)
        x_dot[:3] = L_inv @ omega_b
        x_dot[3:] = np.diag(1/np.diag(self.J)) \
                                @ (- Omega @ self.J @ omega \
                                + 3 * self.n**2 * G @ self.J @ g \
                                + self.M_d + M_c) + self.n*R_dot
        print(t)
        return x_dot

class Controller:

    def __init__(self, K_p, K_d, response_function = "Default",
                 sampling_frequency = np.inf,
                 J = np.identity(3), T_max = np.zeros(3), n = 0, M_max = np.array([np.inf,np.inf,np.inf])):
        self.K_p = K_p
        self.K_d = K_d
        self.e = np.zeros(3, dtype=object)
        self.f = sampling_frequency
        self.J = J
        self.T_max = T_max
        self.n = n
        self.M_max = M_max
        self.u_0 = np.zeros(np.size(T_max))
        self.u = self.u_0
        self.u_1 = self.u_0

        if response_function == "Quaternion_PD":
            self.response = self.PD_response_q
        elif response_function == "Euler_PD":
            self.response = self.PD_response_euler
        elif response_function == "Euler_NDI":
            self.response = self.NDI_response_euler
        elif response_function == "Default" or response_function == "Quaternion_NDI":
            self.response = self.NDI_response_q
        elif response_function == "Quaternion_NDI_TS":
            self.response = self.NDI_TS_response_q
        elif response_function == "Euler_NDI_TS":
            self.response = self.NDI_TS_response_euler
        elif response_function == "Quaternion_INDI_TS":
            self.response = self.INDI_TS_response_q
        elif response_function == "Euler_INDI_TS":
            self.response = self.INDI_TS_response_euler
        else:
            print(f"Unsupported response function: {response_function}")


    def PD_response_q(self, u, x, omega_b, omega_b_dot = 0):

        Omega_b = np.array([[0, -omega_b[2], omega_b[1], -omega_b[0]],
                            [omega_b[2], 0, -omega_b[0], -omega_b[1]],
                            [-omega_b[1], omega_b[0], 0, -omega_b[2]],
                            [omega_b[0], omega_b[1], omega_b[2], 0]])

        L_c = np.array([[ u[3],  u[2], -u[1], -u[0]],
                        [-u[2],  u[3],  u[0], -u[1]],
                        [ u[1], -u[0],  u[3], -u[2]],
                        [ u[0],  u[1],  u[2],  u[3]]])

        e_dot = - 1/2 * Omega_b @ x
        return (self.K_p * (L_c @ x) + self.K_d * e_dot)[:3]

    def PD_response_euler(self, u, x, omega_b, omega_b_dot = 0):

        L_inv = 1/np.cos(x[1]) * np.array([[np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.sin(x[1])],
                                           [0, np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.cos(x[1])],
                                           [0, np.sin(x[0]), np.cos(x[0])]])

        e_dot = L_inv[:3,:3] @ omega_b
        return self.K_p * (x-u) + self.K_d * e_dot

    def NDI_response_euler(self, u, x, omega_b, omega_b_dot = 0):

        C3 = np.array([[np.cos(x[2]), np.sin(x[2]), 0],
                       [-np.sin(x[2]), np.cos(x[2]), 0],
                       [0, 0, 1]])

        C2 = np.array([[np.cos(x[1]), 0, -np.sin(x[1])],
                       [0, 1, 0, ],
                       [np.sin(x[1]), 0, np.cos(x[1])]])

        C1 = np.array([[1, 0, 0, ],
                       [0, np.cos(x[0]), np.sin(x[0])],
                       [0, -np.sin(x[0]), np.cos(x[0])]])

        L_inv = 1 / np.cos(x[1]) * np.array(
            [[np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.sin(x[1])],
             [0, np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.cos(x[1])],
             [0, np.sin(x[0]), np.cos(x[0])]])

        C = C1 @ C2 @ C3

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        Omega_b = np.array([[ 0,         -omega_b[2], omega_b[1]],
                            [ omega_b[2], 0,         -omega_b[0]],
                            [-omega_b[1], omega_b[0], 0         ]])

        C_dot = -Omega_b @ C

        R_dot = np.array([C_dot[0,1], C_dot[1,1], C_dot[2,1]])

        omega = omega_b - self.n * R

        Omega = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])

        A = L_inv @ np.diag(1/np.diag(self.J))

        dL_inv_omega_b_dx = np.zeros((3,3))

        dL_inv_omega_b_dx[0,0] = (np.sin(x[0])*omega_b[1] - np.sin(x[0])*omega_b[2])*np.tan(x[1])
        dL_inv_omega_b_dx[0,1] = (np.sin(x[0])*omega_b[1] + np.cos(x[0])*omega_b[2])/np.cos(x[1])**2
        dL_inv_omega_b_dx[1,0] = -np.sin(x[0])*omega_b[1] - np.cos(x[0])*omega_b[2]
        dL_inv_omega_b_dx[2,0] = (np.cos(x[0])*omega_b[1] - np.sin(x[0])*omega_b[2])/np.cos(x[1])
        dL_inv_omega_b_dx[2,1] = (np.sin(x[0])*omega_b[1] + np.cos(x[0])*omega_b[2])*np.tan(x[1])/np.cos(x[1])

        e_dot = L_inv[:3, :3] @ omega_b

        domega_b_dt = np.diag(1/np.diag(self.J)) \
                                @ (- Omega @ self.J @ omega \
                                + 3 * self.n**2 * G @ self.J @ g) \
                                + self.n*R_dot

        B = np.hstack((dL_inv_omega_b_dx, L_inv)) @ np.hstack((e_dot, domega_b_dt)) + L_inv@np.diag(1/np.diag(self.J))@self.T_max

        return np.linalg.inv(A) @ (self.K_p * (x-u) + self.K_d * e_dot - B)

    def NDI_response_q(self, u, x, omega_b, omega_b_dot = 0):

        Q = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])

        C = (x[3]**2 - x[:3] @ x[:3])*np.identity(3) + 2 * x[:3].reshape((-1,1)) @ x[:3].reshape((1,-1)) - 2 * x[3] * Q

        R = np.array([C[0, 1], C[1, 1], C[2, 1]])

        g = np.array([C[0, 2], C[1, 2], C[2, 2]])

        G = np.array([[0, -C[2, 2], C[1, 2]],
                      [C[2, 2], 0, -C[0, 2]],
                      [-C[1, 2], C[0, 2], 0]])

        Omega_b = np.array([[0, -omega_b[2], omega_b[1], -omega_b[0]],
                            [omega_b[2], 0, -omega_b[0], -omega_b[1]],
                            [-omega_b[1], omega_b[0], 0, -omega_b[2]],
                            [omega_b[0], omega_b[1], omega_b[2], 0]])

        C_dot = -Omega_b[:3, :3] @ C

        R_dot = np.array([C_dot[0, 1], C_dot[1, 1], C_dot[2, 1]])

        omega = omega_b - self.n * R

        Omega = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])

        A = 1/2 * (Q - np.identity(3)*x[3]) @ np.diag(1 / np.diag(self.J))
        e_dot = - 1/2 * Omega_b @ x

        domega_b_dt = np.diag(1 / np.diag(self.J)) \
                      @ (- Omega @ self.J @ omega \
                         + 3 * self.n ** 2 * G @ self.J @ g) \
                      + self.n * R_dot

        B = 1/2 * (np.hstack((-Omega_b[:3, :3], (Q - np.identity(3)*x[3]))) @ np.hstack((e_dot[:3], domega_b_dt)) + (Q - np.identity(3)*x[3]) @ np.diag(
            1 / np.diag(self.J)) @ self.T_max)


        L_c = np.array([[u[3], u[2], -u[1], -u[0]],
                        [-u[2], u[3], u[0], -u[1]],
                        [u[1], -u[0], u[3], -u[2]],
                        [u[0], u[1], u[2], u[3]]])

        return np.linalg.inv(A) @ ((self.K_p * (L_c @ x) + self.K_d * e_dot)[:3] - B)

    def NDI_TS_response_q(self, u, x, omega_b, omega_b_dot = 0):
        L_c = np.array([[u[3], u[2], -u[1], -u[0]],
                        [-u[2], u[3], u[0], -u[1]],
                        [u[1], -u[0], u[3], -u[2]],
                        [u[0], u[1], u[2], u[3]]])

        L = 2 * np.array([[x[3], x[2], -x[1], -x[0]],
                          [-x[2], x[3], x[0], -x[1]],
                          [x[1], -x[0], x[3], -x[2]],
                          [x[0], x[1], x[2], x[3]]])

        Q = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])

        C = (x[3] ** 2 - x[:3] @ x[:3]) * np.identity(3) + 2 * x[:3].reshape((-1, 1)) @ x[:3].reshape((1, -1)) - 2 * x[
            3] * Q

        R = np.array([C[0, 1], C[1, 1], C[2, 1]])

        g = np.array([C[0, 2], C[1, 2], C[2, 2]])

        G = np.array([[0, -C[2, 2], C[1, 2]],
                      [C[2, 2], 0, -C[0, 2]],
                      [-C[1, 2], C[0, 2], 0]])

        Omega_b = np.array([[0, -omega_b[2], omega_b[1], -omega_b[0]],
                            [omega_b[2], 0, -omega_b[0], -omega_b[1]],
                            [-omega_b[1], omega_b[0], 0, -omega_b[2]],
                            [omega_b[0], omega_b[1], omega_b[2], 0]])

        C_dot = -Omega_b[:3, :3] @ C

        R_dot = np.array([C_dot[0, 1], C_dot[1, 1], C_dot[2, 1]])

        omega = omega_b - self.n * R

        Omega = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])

        u1 = (L @ (self.K_p * (L_c @ x)))[:3]

        B = np.diag(1 / np.diag(self.J)) \
                      @ (- Omega @ self.J @ omega
                         + 3 * self.n ** 2 * G @ self.J @ g
                         + self.T_max) \
                         + self.n * R_dot

        return self.J @ (self.K_d[:3] * (omega_b - u1) - B)

    def NDI_TS_response_euler(self, u, x, omega_b, omega_b_dot = 0):

        L = np.array([[1, 0, 0],
                      [0, np.cos(x[0]), np.sin(x[0])*np.cos(x[1])],
                      [0, -np.sin(x[0]), np.cos(x[0])*np.cos(x[1])]])

        C3 = np.array([[np.cos(x[2]), np.sin(x[2]), 0],
                       [-np.sin(x[2]), np.cos(x[2]), 0],
                       [0, 0, 1]])

        C2 = np.array([[np.cos(x[1]), 0, -np.sin(x[1])],
                       [0, 1, 0, ],
                       [np.sin(x[1]), 0, np.cos(x[1])]])

        C1 = np.array([[1, 0, 0, ],
                       [0, np.cos(x[0]), np.sin(x[0])],
                       [0, -np.sin(x[0]), np.cos(x[0])]])

        C = C1 @ C2 @ C3

        R = np.array([C[0,1], C[1,1], C[2,1]])

        g = np.array([C[0,2], C[1,2], C[2,2]])

        G = np.array([[0,       -C[2,2], C[1,2]],
                      [C[2,2],  0,      -C[0,2]],
                      [-C[1,2], C[0,2],  0     ]])

        Omega_b = np.array([[ 0,         -omega_b[2], omega_b[1]],
                            [ omega_b[2], 0,         -omega_b[0]],
                            [-omega_b[1], omega_b[0], 0         ]])

        C_dot = -Omega_b @ C

        R_dot = np.array([C_dot[0,1], C_dot[1,1], C_dot[2,1]])

        omega = omega_b - self.n * R

        Omega = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])

        u1 = L @ (self.K_p * (x - u))

        B = np.diag(1 / np.diag(self.J)) \
            @ (- Omega @ self.J @ omega
               + 3 * self.n ** 2 * G @ self.J @ g
               + self.T_max) \
               + self.n * R_dot

        return self.J @ (self.K_d * (omega_b - u1) - B)

    def INDI_TS_response_q(self, u, x, omega_b, omega_b_dot):

        L_c = np.array([[u[3], u[2], -u[1], -u[0]],
                        [-u[2], u[3], u[0], -u[1]],
                        [u[1], -u[0], u[3], -u[2]],
                        [u[0], u[1], u[2], u[3]]])

        L = 2 * np.array([[x[3], x[2], -x[1], -x[0]],
                          [-x[2], x[3], x[0], -x[1]],
                          [x[1], -x[0], x[3], -x[2]],
                          [x[0], x[1], x[2], x[3]]])

        Omega_b = np.array([[0, -omega_b[2], omega_b[1], -omega_b[0]],
                            [omega_b[2], 0, -omega_b[0], -omega_b[1]],
                            [-omega_b[1], omega_b[0], 0, -omega_b[2]],
                            [omega_b[0], omega_b[1], omega_b[2], 0]])

        self.u += (L @ (self.K_p * (L_c @ x) + 1/2 * Omega_b @ x))[:3]

        self.u_1 += self.J @ (self.K_d[:3] * (omega_b - self.u) - omega_b_dot)
        return self.u_1

    def INDI_TS_response_euler(self, u, x, omega_b, omega_b_dot):

        L = np.array([[1, 0, 0],
                      [0, np.cos(x[0]), np.sin(x[0]) * np.cos(x[1])],
                      [0, -np.sin(x[0]), np.cos(x[0]) * np.cos(x[1])]])

        L_inv = 1 / np.cos(x[1]) * np.array(
            [[np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.sin(x[1])],
             [0, np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.cos(x[1])],
             [0, np.sin(x[0]), np.cos(x[0])]])

        self.u += L @ (self.K_p * (x - u) - L_inv @ omega_b)

        self.u_1 += self.J @ (self.K_d[:3] * (omega_b - self.u) - omega_b_dot)
        return self.u_1

    def reset(self):
        self.u = self.u_0
        self.u_1 = self.u_0

