"""
cbf.py - Control Barrier Function core math

Implements Zhang et al. 2020 CBF haptic teleoperation.
Pure Python - no ROS dependencies, testable standalone.

CBF for unicycle robot (LIMO):
    State: (x, y, theta)
    Control: u = (v, omega) - linear and angular velocity
    Barrier: b = d - r_safe (distance to obstacle minus safety radius)

CBF Constraint (relative degree 1):
    b_dot + gamma * b >= 0
    => -v*cos(alpha) + gamma*(d - r_safe) >= 0
    => v*cos(alpha) <= gamma*(d - r_safe)

Zhang 2020 haptic force law:
    F = Kf * (u_safe - u_ref)
    Force is zero when user command is already safe.
    Force guides user toward nearest safe command when unsafe.

Closed-form KKT solution avoids needing a QP solver library.
"""

import numpy as np


# Camera intrinsics for Astra camera on LIMO
# Used to convert pixel coords -> bearing angle
FOCAL_X = 570.0   # focal length in pixels (x)
CENTER_X = 320.0  # principal point x (image center)
CENTER_Y = 200.0  # principal point y (adjusted for 400px height)


class CBFController:
    """
    CBF-QP controller for safe teleoperation.

    Solves:
        min  (v - v_ref)^2 + w_omega*(omega - omega_ref)^2
        s.t. a*v <= c  (CBF safety constraint)
             v_min <= v <= v_max
             omega_min <= omega <= omega_max

    Has a closed-form solution via KKT conditions.
    """

    def __init__(self,
                 r_safe=0.4,          # safety radius in meters
                 gamma=1.5,           # CBF class-K gain (higher = more aggressive braking)
                 kf=10.0,             # haptic force gain (N per m/s of velocity diff)
                 w_omega=0.5,         # weight on omega in QP (lower = steering adjusts more freely)
                 v_max=0.3,           # max forward velocity (m/s)
                 v_min=-0.2,          # max reverse velocity (m/s)
                 omega_max=1.0,       # max angular velocity (rad/s)
                 f_max=2.5):          # max haptic force magnitude (N)
        self.r_safe = r_safe
        self.gamma = gamma
        self.kf = kf
        self.w_omega = w_omega
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.f_max = f_max

    def pixel_to_bearing(self, pixel_x):
        """
        Convert pixel x coordinate to bearing angle (radians).

        Args:
            pixel_x: x coordinate in image (0 = left, 640 = right)

        Returns:
            alpha: bearing angle in radians
                   negative = obstacle to the left
                   zero = obstacle straight ahead
                   positive = obstacle to the right
        """
        # Standard pinhole camera model
        # angle = arctan((pixel_x - cx) / fx)
        return np.arctan2(pixel_x - CENTER_X, FOCAL_X)

    def compute_barrier(self, depth, pixel_x):
        """
        Compute CBF value b and constraint coefficients.

        b = d - r_safe
        b_dot = -v * cos(alpha)  (approach velocity)

        CBF constraint: b_dot + gamma*b >= 0
        => -v*cos(alpha) + gamma*(d - r_safe) >= 0
        => v*cos(alpha) <= gamma*(d - r_safe)
        => a*v <= c   where a=cos(alpha), c=gamma*(d-r_safe)

        Args:
            depth: distance to obstacle in meters
            pixel_x: pixel x coordinate of obstacle

        Returns:
            b: barrier value (positive = safe, negative = unsafe)
            a: CBF constraint coefficient for v
            c: CBF constraint right-hand side
        """
        b = depth - self.r_safe

        alpha = self.pixel_to_bearing(pixel_x)

        # cos(alpha) is how much forward velocity approaches the obstacle
        # cos(0) = 1.0 → obstacle dead ahead, full braking needed
        # cos(pi/2) = 0.0 → obstacle to the side, no constraint on v
        a = np.cos(alpha)

        # RHS of constraint: how much closing speed is allowed
        c = self.gamma * b

        return b, a, c

    def solve_qp(self, v_ref, omega_ref, a, c):
        """
        Solve CBF-QP using closed-form KKT conditions.

        QP:
            min  (v - v_ref)^2 + w*(omega - omega_ref)^2
            s.t. a*v <= c              (CBF constraint)
                 v_min <= v <= v_max
                 -omega_max <= omega <= omega_max

        When constraint is already satisfied: u_safe = u_ref (no modification)
        When violated: project u_ref onto constraint boundary

        KKT closed-form (single linear constraint, diagonal weight matrix):
            If a*v_ref <= c: u_safe = u_ref
            Else: v_safe = (c + a*(v_ref - c/a)) / ... (see derivation below)

        Args:
            v_ref: desired linear velocity (m/s)
            omega_ref: desired angular velocity (rad/s)
            a: CBF constraint coefficient
            c: CBF constraint RHS

        Returns:
            v_safe: safe linear velocity
            omega_safe: safe angular velocity
        """
        # Clip user input to hardware limits first
        v_ref = np.clip(v_ref, self.v_min, self.v_max)
        omega_ref = np.clip(omega_ref, -self.omega_max, self.omega_max)

        # Check if user command already satisfies CBF constraint
        if a * v_ref <= c:
            # Already safe - no modification needed
            return v_ref, omega_ref

        # Constraint violated - project onto safety boundary
        # We only constrain v (omega is not in the constraint for this barrier)
        # v_safe = c / a  would be exact constraint satisfaction
        # But we use KKT projection to stay as close to v_ref as possible

        # KKT conditions give: v_safe = v_ref - lambda * a
        # where lambda = (a*v_ref - c) / a^2
        # So: v_safe = v_ref - (a*v_ref - c)/a = c/a ... same result
        # Because our constraint only involves v (b_coeff for omega = 0)

        if abs(a) < 1e-6:
            # Obstacle is perpendicular - no constraint on forward velocity
            v_safe = v_ref
        else:
            # Project v onto constraint: a*v = c (boundary)
            v_safe = c / a

        # Clip to hardware limits
        v_safe = np.clip(v_safe, self.v_min, self.v_max)
        omega_safe = np.clip(omega_ref, -self.omega_max, self.omega_max)

        return v_safe, omega_safe

    def compute_haptic_force(self, v_ref, omega_ref, v_safe, omega_safe):
        """
        Compute haptic force feedback (Zhang 2020 Eq. 6):
            F = Kf * (u_safe - u_ref)

        Force is:
            - Zero when user command is already safe
            - Nonzero when CBF modified the command
            - Points toward the nearest safe command
            - Direction tells user HOW to correct their input

        Args:
            v_ref, omega_ref: user's desired command
            v_safe, omega_safe: CBF-corrected safe command

        Returns:
            f_linear: force in linear direction (N) - negative = slow down
            f_angular: force in angular direction (N) - tells user to steer
        """
        f_linear = self.kf * (v_safe - v_ref)
        f_angular = self.kf * (omega_safe - omega_ref)

        # Clamp total force magnitude to hardware limit
        f_mag = np.sqrt(f_linear**2 + f_angular**2)
        if f_mag > self.f_max:
            scale = self.f_max / f_mag
            f_linear *= scale
            f_angular *= scale

        return f_linear, f_angular

    def step(self, v_ref, omega_ref, depth, pixel_x):
        """
        Full CBF step: given user command and obstacle info,
        compute safe command and haptic force.

        Args:
            v_ref: user's desired linear velocity (m/s)
            omega_ref: user's desired angular velocity (rad/s)
            depth: distance to closest obstacle (meters)
            pixel_x: pixel x coordinate of closest obstacle

        Returns:
            v_safe: safe linear velocity to send to robot
            omega_safe: safe angular velocity to send to robot
            f_linear: haptic force - linear channel (N)
            f_angular: haptic force - angular channel (N)
            b: barrier value (positive=safe, negative=unsafe)
        """
        # Compute barrier and constraint
        b, a, c = self.compute_barrier(depth, pixel_x)

        # Solve CBF-QP
        v_safe, omega_safe = self.solve_qp(v_ref, omega_ref, a, c)

        # Compute haptic force
        f_linear, f_angular = self.compute_haptic_force(
            v_ref, omega_ref, v_safe, omega_safe
        )

        return v_safe, omega_safe, f_linear, f_angular, b


# ============================================================
# STANDALONE TEST
# python3 cbf.py
# ============================================================
if __name__ == '__main__':
    print("=== CBF Standalone Test ===\n")
    cbf = CBFController(r_safe=0.4, gamma=1.5, kf=10.0)

    # Test cases
    test_cases = [
        # (v_ref, omega_ref, depth, pixel_x, description)
        (0.3, 0.0, 2.0, 320, "Far obstacle ahead - should be unchanged"),
        (0.3, 0.0, 0.5, 320, "Close obstacle ahead - should brake"),
        (0.3, 0.0, 0.3, 320, "At safety radius - should stop"),
        (0.3, 0.0, 0.2, 320, "Inside safety radius - should stop/reverse"),
        (0.3, 0.0, 0.5, 100, "Close but to the left - less braking"),
        (0.0, 0.5, 0.5, 320, "Rotating in place near obstacle - safe"),
    ]

    for v_ref, omega_ref, depth, px, desc in test_cases:
        v_safe, omega_safe, f_lin, f_ang, b = cbf.step(
            v_ref, omega_ref, depth, px
        )
        print(f"{desc}")
        print(f"  Input:  v={v_ref:.2f} m/s, ω={omega_ref:.2f} rad/s")
        print(f"  Depth:  {depth:.2f}m, b={b:.3f}")
        print(f"  Output: v={v_safe:.2f} m/s, ω={omega_safe:.2f} rad/s")
        print(f"  Force:  F_lin={f_lin:.2f}N, F_ang={f_ang:.2f}N")
        print()