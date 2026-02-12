import numpy as np
from typing import Tuple


class QuadPendEnv:
    """Quadcopter-pendulum environment: numpy dynamics, control clipping."""

    def __init__(self, param_dict: dict = None):
        if param_dict is None:
            import os, sys
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from create_arg_parser import create_arg_parser
            from problems.quad_pend import create_quad_pend_param_dict
            args = create_arg_parser().parse_known_args()[0]
            self.param_dict = create_quad_pend_param_dict(args)
        else:
            self.param_dict = param_dict

        self.__dict__.update(self.param_dict)

        state_names = [
            "gamma", "beta", "alpha", "dgamma", "dbeta", "dalpha",
            "phi", "theta", "dphi", "dtheta", "x", "y", "z", "dx", "dy", "dz",
        ]
        self.i = dict(zip(state_names, range(len(state_names))))
        self.dt = 0.00005
        self.g = 9.81

        self.control_lim_verts = self._compute_control_lim_vertices()

    def _compute_control_lim_vertices(self) -> np.ndarray:
        """Pre-computes the 16 vertices of the control-limit set in force space."""
        M = np.array([
            [self.k1, self.k1, self.k1, self.k1],
            [0, -self.l * self.k1, 0, self.l * self.k1],
            [self.l * self.k1, 0, -self.l * self.k1, 0],
            [-self.k2, self.k2, -self.k2, self.k2],
        ])
        self.mixer = M
        self.mixer_inv = np.linalg.inv(M)

        # 16 vertices of the unit hypercube [0,1]^4 (impulse space)
        r1 = np.concatenate((np.zeros(8), np.ones(8)))
        r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
        r3 = np.concatenate((np.zeros(2), np.ones(2)) * 4)
        r4 = np.zeros(16); r4[1::2] = 1.0
        impulse_verts = np.stack([r1, r2, r3, r4])  # (4, 16)

        force_verts = M @ impulse_verts - np.array([[self.M * self.g], [0.0], [0.0], [0.0]])
        return force_verts.T.astype("float32")

    def _f_model(self, x: np.ndarray) -> np.ndarray:
        """Drift term f(x) of the control-affine dynamics ẋ = f(x) + g(x)u."""
        if x.ndim == 1:
            x = x[None]
        bs = x.shape[0]

        gamma = x[:, self.i["gamma"]]
        beta  = x[:, self.i["beta"]]
        alpha = x[:, self.i["alpha"]]
        phi   = x[:, self.i["phi"]]
        theta = x[:, self.i["theta"]]
        dphi  = x[:, self.i["dphi"]]
        dtheta = x[:, self.i["dtheta"]]

        # Rotation matrix (ZYX Euler: alpha, beta, gamma)
        R = np.zeros((bs, 3, 3))
        R[:, 0, 0] = np.cos(alpha) * np.cos(beta)
        R[:, 0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
        R[:, 0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
        R[:, 1, 0] = np.sin(alpha) * np.cos(beta)
        R[:, 1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
        R[:, 1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
        R[:, 2, 0] = -np.sin(beta)
        R[:, 2, 1] = np.cos(beta) * np.sin(gamma)
        R[:, 2, 2] = np.cos(beta) * np.cos(gamma)

        k_x, k_y, k_z = R[:, 0, 2], R[:, 1, 2], R[:, 2, 2]

        ddphi = (
            3.0 * (k_y * np.cos(phi) + k_z * np.sin(phi)) * (self.M * self.g)
            / (2 * self.M * self.L_p * np.cos(theta))
            + 2 * dtheta * dphi * np.tan(theta)
        )
        ddtheta = (
            3.0 * (-k_x * np.cos(theta) - k_y * np.sin(phi) * np.sin(theta)
                   + k_z * np.cos(phi) * np.sin(theta))
            * (self.M * self.g) / (2.0 * self.M * self.L_p)
            - np.square(dphi) * np.sin(theta) * np.cos(theta)
        )
        ddx = k_x * self.g
        ddy = k_y * self.g
        ddz = k_z * self.g - self.g

        f = np.vstack([
            x[:, self.i["dgamma"]], x[:, self.i["dbeta"]], x[:, self.i["dalpha"]],
            np.zeros(bs), np.zeros(bs), np.zeros(bs),
            dphi, dtheta, ddphi, ddtheta,
            x[:, self.i["dx"]], x[:, self.i["dy"]], x[:, self.i["dz"]],
            ddx, ddy, ddz,
        ]).T
        return f

    def _g_model(self, x: np.ndarray) -> np.ndarray:
        """Control-input matrix g(x) of the dynamics ẋ = f(x) + g(x)u."""
        if x.ndim == 1:
            x = x[None]
        bs = x.shape[0]

        gamma = x[:, self.i["gamma"]]
        beta  = x[:, self.i["beta"]]
        alpha = x[:, self.i["alpha"]]
        phi   = x[:, self.i["phi"]]
        theta = x[:, self.i["theta"]]

        R = np.zeros((bs, 3, 3))
        R[:, 0, 0] = np.cos(alpha) * np.cos(beta)
        R[:, 0, 1] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
        R[:, 0, 2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
        R[:, 1, 0] = np.sin(alpha) * np.cos(beta)
        R[:, 1, 1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
        R[:, 1, 2] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
        R[:, 2, 0] = -np.sin(beta)
        R[:, 2, 1] = np.cos(beta) * np.sin(gamma)
        R[:, 2, 2] = np.cos(beta) * np.cos(gamma)

        k_x, k_y, k_z = R[:, 0, 2], R[:, 1, 2], R[:, 2, 2]

        J_inv = np.diag([1.0 / self.J_x, 1.0 / self.J_y, 1.0 / self.J_z])
        dd_drone_angles = R @ J_inv  # (bs, 3, 3)

        ddphi   = 3.0 * (k_y * np.cos(phi) + k_z * np.sin(phi)) / (2 * self.M * self.L_p * np.cos(theta))
        ddtheta = 3.0 * (-k_x * np.cos(theta) - k_y * np.sin(phi) * np.sin(theta)
                         + k_z * np.cos(phi) * np.sin(theta)) / (2.0 * self.M * self.L_p)

        g = np.zeros((bs, 16, 4))
        g[:, 3:6, 1:] = dd_drone_angles
        g[:, 8,  0]   = ddphi
        g[:, 9,  0]   = ddtheta
        g[:, 13:, 0]  = (1.0 / self.M) * np.array([k_x, k_y, k_z]).T

        return g

    def x_dot_open_loop_model(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes ẋ = f(x) + g(x)u using the model parameters."""
        if u.ndim == 1:
            u = u[None]
        if x.ndim == 1:
            x = x[None]
        u_clamped, _ = self.clip_u(u)
        return self._f_model(x) + (self._g_model(x) @ u_clamped[:, :, None])[:, :, 0]

    def x_dot_open_loop(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Alias for x_dot_open_loop_model (no mismatch / noise)."""
        return self.x_dot_open_loop_model(x, u)

    def _smooth_clamp(self, motor_impulses: np.ndarray) -> np.ndarray:
        return np.clip(motor_impulses, 0, 1)

    def clip_u(self, u: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Clamps u to the motor-saturation limits via the mixer matrix."""
        if u.ndim == 1:
            u = u[None]
        u_gc = u + np.array([self.M * self.g, 0, 0, 0])   # gravity-compensated
        motor_impulses = u_gc @ self.mixer_inv.T
        clamped = self._smooth_clamp(motor_impulses)
        u_clamped = clamped @ self.mixer.T - np.array([self.M * self.g, 0, 0, 0])
        return u_clamped, {"motor_impulses": motor_impulses, "smooth_clamped_motor_impulses": clamped}
