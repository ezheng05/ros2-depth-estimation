import numpy as np

FOCAL_X = 570.0
CENTER_X = 320.0


class CBFController:
    def __init__(self, r_safe=0.3, gamma=0.8, kf=0.5, w_omega=0.5,
                 v_max=0.3, v_min=-0.2, omega_max=1.0, f_max=0.3):
        self.r_safe = r_safe
        self.gamma = gamma
        self.kf = kf
        self.w_omega = w_omega
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.f_max = f_max

    def pixel_to_bearing(self, px):
        return np.arctan2(px - CENTER_X, FOCAL_X)

    def barrier(self, depth, px):
        b = depth - self.r_safe
        a = np.cos(self.pixel_to_bearing(px))
        c = self.gamma * b
        return b, a, c

    def solve(self, v_ref, w_ref, a, c):
        v_ref = np.clip(v_ref, self.v_min, self.v_max)
        w_ref = np.clip(w_ref, -self.omega_max, self.omega_max)

        if a * v_ref <= c:
            return v_ref, w_ref

        v = c / a if abs(a) > 1e-6 else v_ref
        v = np.clip(v, self.v_min, self.v_max)
        return v, w_ref

    def force(self, v_ref, w_ref, v_safe, w_safe):
        fl = self.kf * (v_safe - v_ref)
        fa = self.kf * (w_safe - w_ref)

        mag = np.sqrt(fl**2 + fa**2)
        if mag > self.f_max:
            s = self.f_max / mag
            fl *= s
            fa *= s
        return fl, fa

    def step(self, v_ref, w_ref, depth, px):
        b, a, c = self.barrier(depth, px)
        vs, ws = self.solve(v_ref, w_ref, a, c)
        fl, fa = self.force(v_ref, w_ref, vs, ws)
        return vs, ws, fl, fa, b


if __name__ == '__main__':
    cbf = CBFController()
    cases = [
        (0.3, 0.0, 2.0, 320, "far ahead"),
        (0.3, 0.0, 0.5, 320, "close ahead"),
        (0.3, 0.0, 0.3, 320, "at r_safe"),
        (0.3, 0.0, 0.2, 320, "inside r_safe"),
        (0.3, 0.0, 0.5, 100, "close left"),
        (0.0, 0.5, 0.5, 320, "rotate near obstacle"),
    ]
    for vr, wr, d, px, desc in cases:
        vs, ws, fl, fa, b = cbf.step(vr, wr, d, px)
        print(f"{desc}: v={vr:.2f}->{vs:.2f} b={b:.3f} F=({fl:.2f},{fa:.2f})")
