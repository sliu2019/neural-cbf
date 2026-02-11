"""CBF safety filter for the quadcopter-pendulum system.

The controller computes a minimal-intervention safe control input by solving a
quadratic program (QP) that enforces the CBF constraint
    ∇φ*(x)·(f(x) + g(x)u) ≤ −ε
whenever the system is on or outside the CBF zero-level set.

Three operating modes (determined each step by the current φ*(x) value):
  * Inside  (φ* < 0, and next φ* < 0): apply the reference control as-is.
  * On boundary (φ* < 0, but next φ* ≥ 0): solve QP with ε = eps_bdry.
  * Outside (φ* > 0): solve QP with ε = eps_outside.
"""
import math

import numpy as np
from cvxopt import matrix, solvers
import control

solvers.options["show_progress"] = False

g = 9.81


class CBFController:
    """Quadratic-program-based CBF safety filter.

    Args:
        env: QuadPendEnv instance providing dynamics (_f_model, _g_model) and dt.
        cbf_obj: PhiNumpy instance with phi_star_fn(x) and phi_grad(x) methods.
        param_dict: Problem parameter dictionary (u_dim, k1, k2, l, M, …).
        args: Namespace with rollout_u_ref, rollout_LQR_q, rollout_LQR_r.
        eps_bdry: CBF slack penalty when on the boundary.
        eps_outside: CBF slack penalty when outside the safe set.
    """

    def __init__(self, env, cbf_obj, param_dict: dict, args,
                 eps_bdry: float = 1.0, eps_outside: float = 5.0):
        # Store all constructor args as attributes and expand param_dict entries
        self.env = env
        self.cbf_obj = cbf_obj
        self.param_dict = param_dict
        self.args = args
        self.eps_bdry = eps_bdry
        self.eps_outside = eps_outside
        self.__dict__.update(param_dict)   # exposes k1, k2, l, M, u_dim, …

        # Pre-compute the mixer matrix mapping motor thrusts → [F, τx, τy, τz]
        self.mixer = np.array([
            [self.k1,          self.k1,           self.k1,          self.k1         ],
            [0,               -self.l * self.k1,  0,                 self.l * self.k1],
            [self.l * self.k1, 0,                -self.l * self.k1,  0               ],
            [-self.k2,         self.k2,          -self.k2,           self.k2         ],
        ])

        # Optionally compute an LQR reference controller
        if self.args.rollout_u_ref == "LQR":
            L_p = param_dict["L_p"]
            M   = param_dict["M"]
            J_x = param_dict["J_x"]
            J_y = param_dict["J_y"]
            J_z = param_dict["J_z"]

            # Linearised dynamics around hover (rotational + pendulum states only)
            A = np.zeros((10, 10))
            A[0:3, 3:6] = np.eye(3)
            A[6:8, 8:10] = np.eye(2)
            A[8, 0]  = -3 * g / (2 * L_p)
            A[9, 1]  = -3 * g / (2 * L_p)
            A[8, 6]  =  3 * g / (2 * L_p)
            A[9, 7]  =  3 * g / (2 * L_p)

            B = np.zeros((10, 4))
            B[3:6, 1:4] = np.diag([1.0 / J_x, 1.0 / J_y, 1.0 / J_z])

            q = self.args.rollout_LQR_q
            r = self.args.rollout_LQR_r
            K, _, _ = control.lqr(A, B, q * np.eye(10), r * np.eye(4))
            self.K = K

    # ------------------------------------------------------------------
    # Reference control
    # ------------------------------------------------------------------

    def compute_u_ref(self, t: int, x: np.ndarray) -> np.ndarray:
        """Returns the (unconstrained) reference control input."""
        if self.args.rollout_u_ref == "unactuated":
            return np.zeros(self.u_dim)
        elif self.args.rollout_u_ref == "LQR":
            return -self.K @ np.squeeze(x)[:10]
        else:
            raise ValueError("Unknown rollout_u_ref: %s" % self.args.rollout_u_ref)

    # ------------------------------------------------------------------
    # Main control interface
    # ------------------------------------------------------------------

    def compute_control(self, t: int, x: np.ndarray):
        """Computes a CBF-filtered safe control input.

        Args:
            t: Current timestep index.
            x: Current state (1, 16) numpy array.

        Returns:
            u: Safe control input (u_dim,).
            debug_dict: Dict with keys apply_u_safe, u_ref, phi_vals, qp_slack,
                        qp_rhs, qp_lhs, impulses, inside_boundary,
                        on_boundary, outside_boundary, dist_between_xs,
                        phi_grad_mag, phi_grad.
        """
        u_ref = self.compute_u_ref(t, x)
        phi_vals = self.cbf_obj.phi_star_fn(x)   # (1, r+1)
        phi_grad = self.cbf_obj.phi_grad(x)       # (1, 16)

        # Predict next state under reference control to detect boundary crossing
        x_next = x + self.env.dt * self.env.x_dot_open_loop_model(x, u_ref)
        next_phi_val = self.cbf_obj.phi_star_fn(x_next)

        dist_between_xs = float(np.linalg.norm(x_next - x))
        phi_grad_mag = float(np.linalg.norm(phi_grad))

        # Determine operating mode
        if phi_vals[0, -1] > 0:
            eps = self.eps_outside
            apply_u_safe = True
            inside_boundary, on_boundary, outside_boundary = False, False, True
        elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:
            eps = self.eps_bdry
            apply_u_safe = True
            inside_boundary, on_boundary, outside_boundary = False, True, False
        else:
            # Safely inside: skip QP
            return u_ref, {
                "apply_u_safe": False, "u_ref": u_ref,
                "phi_vals": phi_vals.flatten(), "qp_slack": None,
                "qp_rhs": None, "qp_lhs": None, "impulses": None,
                "inside_boundary": True, "on_boundary": False, "outside_boundary": False,
                "dist_between_xs": dist_between_xs, "phi_grad_mag": phi_grad_mag,
                "phi_grad": phi_grad,
            }

        # ------------------------------------------------------------------
        # Build and solve the QP
        # Optimisation variables: [u (4), impulses (4), slack (1)]  → 9 total
        #
        # Objective: min ||u - u_ref||² + w·slack
        # Constraints:
        #   CBF:  ∇φ·g·u - slack ≤ -∇φ·f - ε
        #   Mixer: -u + mixer·impulses = [Mg, 0, 0, 0]ᵀ  (equality)
        #   Impulse bounds: 0 ≤ impulses ≤ 1
        #   slack ≥ 0
        # ------------------------------------------------------------------
        f_x = np.reshape(self.env._f_model(x), (16, 1))
        g_x = self.env._g_model(x)                         # (16, 4)
        phi_grad_col = np.reshape(phi_grad, (16, 1))

        lhs = phi_grad_col.T @ g_x   # (1, 4)
        rhs = float(-phi_grad_col.T @ f_x - eps)

        w = 1000.0  # slack penalty weight

        P = np.zeros((9, 9))
        P[:4, :4] = 2 * np.eye(4)

        q_vec = np.zeros((9, 1))
        q_vec[:4, 0] = -2 * u_ref
        q_vec[-1, 0] = w

        # Inequality G·x ≤ ρ
        G = np.zeros((10, 9))
        G[0, :4]  = lhs
        G[0, -1]  = -1.0                  # CBF constraint (with slack)
        G[1:5, 4:8] = -np.eye(4)          # impulses ≥ 0
        G[5:9, 4:8] =  np.eye(4)          # impulses ≤ 1
        G[9, -1]  = -1.0                  # slack ≥ 0

        rho = np.zeros((10, 1))
        rho[0, 0] = rhs
        rho[5:9, 0] = 1.0

        # Equality A·x = b  (mixer constraint: motor thrusts → body forces)
        A_eq = np.zeros((4, 9))
        A_eq[:4, :4] = -np.eye(4)
        A_eq[:4, 4:8] = self.mixer
        b_eq = np.array([self.M * g, 0, 0, 0])[:, None]

        try:
            sol = solvers.qp(
                matrix(P), matrix(q_vec),
                matrix(G), matrix(rho),
                matrix(A_eq), matrix(b_eq),
            )
        except Exception as exc:
            print("QP solve failed: %s" % exc)
            raise

        sol_var = np.array(sol["x"])
        u_safe = np.reshape(sol_var[:4], (4,))
        impulses = sol_var[4:8]
        qp_slack = sol_var[-1]

        return u_safe, {
            "apply_u_safe": apply_u_safe, "u_ref": u_ref,
            "phi_vals": phi_vals.flatten(), "qp_slack": qp_slack,
            "qp_rhs": rhs, "qp_lhs": lhs, "impulses": impulses,
            "inside_boundary": inside_boundary, "on_boundary": on_boundary,
            "outside_boundary": outside_boundary,
            "dist_between_xs": dist_between_xs, "phi_grad_mag": phi_grad_mag,
            "phi_grad": phi_grad.flatten(),
        }
