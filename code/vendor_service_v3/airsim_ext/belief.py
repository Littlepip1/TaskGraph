from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


@dataclass
class TargetBelief2D:
    """Constant-velocity Gaussian belief over [x, y, vx, vy]."""

    mean: np.ndarray
    cov: np.ndarray
    process_accel_std: float = 1.2

    @classmethod
    def broad(cls, xy=(0.0, 0.0), pos_std=20.0, vel_std=6.0, process_accel_std=1.2):
        mean = np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float64)
        cov = np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2]).astype(np.float64)
        return cls(mean, cov, process_accel_std)

    def copy(self) -> "TargetBelief2D":
        return TargetBelief2D(self.mean.copy(), self.cov.copy(), self.process_accel_std)

    @staticmethod
    def transition(dt: float) -> np.ndarray:
        return np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def process_cov(self, dt: float) -> np.ndarray:
        # White acceleration model in x/y.
        q = self.process_accel_std**2
        b = np.array([[0.5 * dt**2], [dt]], dtype=np.float64)
        q1 = q * (b @ b.T)
        out = np.zeros((4, 4), dtype=np.float64)
        out[np.ix_([0, 2], [0, 2])] = q1
        out[np.ix_([1, 3], [1, 3])] = q1
        return out

    def predict(self, dt: float) -> None:
        f = self.transition(dt)
        self.mean = f @ self.mean
        self.cov = _sym(f @ self.cov @ f.T + self.process_cov(dt))

    def update_position(self, z_xy: np.ndarray, r_xy: np.ndarray) -> None:
        h = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        z = np.asarray(z_xy, dtype=np.float64).reshape(2)
        r = np.asarray(r_xy, dtype=np.float64).reshape(2, 2)
        innovation = z - h @ self.mean
        s = h @ self.cov @ h.T + r
        k = self.cov @ h.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ innovation
        eye = np.eye(4, dtype=np.float64)
        # Joseph form is a little more stable for repeated anisotropic updates.
        a = eye - k @ h
        self.cov = _sym(a @ self.cov @ a.T + k @ r @ k.T)

    def projected_position_cov(self, steps: int, dt: float) -> np.ndarray:
        b = self.copy()
        for _ in range(max(0, int(steps))):
            b.predict(dt)
        return b.cov[:2, :2].copy()


def delayed_measurement_to_now(z_xy: np.ndarray, source_tick: int, now_tick: int,
                               dt: float, velocity_xy: np.ndarray,
                               r_xy: np.ndarray, process_accel_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Project a stale position measurement to the current tick.

    This is deliberately simple and deployment-friendly: use the receiver's current
    target-velocity estimate to translate the old position and inflate covariance for
    elapsed process uncertainty. It avoids hidden-state access.
    """
    lag = max(0, int(now_tick) - int(source_tick))
    tau = lag * float(dt)
    z_now = np.asarray(z_xy, dtype=np.float64).reshape(2) + tau * np.asarray(velocity_xy, dtype=np.float64).reshape(2)
    q_pos = (0.5 * tau**2 * float(process_accel_std))**2
    r_now = np.asarray(r_xy, dtype=np.float64).reshape(2, 2) + np.eye(2) * q_pos
    return z_now, r_now
