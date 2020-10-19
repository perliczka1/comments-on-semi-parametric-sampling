import numpy as np


class LinearGaussianSampling:
    """
    From the paper:
    Thompson Sampling for Contextual Bandits with Linear Payoffs
    http://proceedings.mlr.press/v28/agrawal13.pdf
    """
    def __init__(self, arms_nb: int, d: int, X: np.array, v: float, seed: int):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)

        self.N = arms_nb
        self.d = d
        self.X = X
        self.v = v

        self.B = np.eye(d)
        self.f = np.zeros((d, 1))
        self.mi_dashed = np.zeros((d, 1))

        self.mi = None
        self.exp_reward = None

    def _update_exp_reward(self) -> np.array:
        self.exp_reward = self.mi.T @ self.X

    def _estimate_mi_cov(self) -> np.array:
        return self.v ** 2 * np.linalg.inv(self.B)

    def _sample_mi(self) -> None:
        mean = self.mi_dashed
        cov = self._estimate_mi_cov()
        mi_flat = self.generator.multivariate_normal(mean.flatten(), cov)
        self.mi = np.expand_dims(mi_flat, axis=1)

    def observe_reward(self, i: int, r: float):
        X_i = np.take(self.X, [i], 1)
        self.B += X_i @ X_i.T
        self.f += X_i * r
        self.mi_dashed = np.linalg.inv(self.B) @ self.f

        self.mi = None
        self.exp_reward = None

    def _get_best_arm(self):
        return np.argmax(self.exp_reward)

    def choose_arm(self):
        self._sample_mi()
        self._update_exp_reward()
        return self._get_best_arm()


