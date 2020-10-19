import numpy as np

class GaussianPriorsSampling:
    """
    From the paper:
    Near-optimal Regret Bounds for Thompson Sampling
    http://www.columbia.edu/~sa3305/j3.pdf
    """
    def __init__(self, arms_nb: int, seed: int):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.N = arms_nb
        self.k = np.zeros(self.N)
        self.mi = np.zeros(self.N)
        self.theta = None # shape N

    def _sample_theta(self) -> None:
        self.theta = self.generator.normal(self.mi, (self.k + 1) ** -0.5)

    def observe_reward(self, i: int, r: float):
        self.mi[i] = (self.mi[i] * (self.k[i] + 1) + r) / (self.k[i] + 2)
        self.k[i] += 1
        self.theta = None

    def _get_best_arm(self):
        return np.argmax(self.theta)

    def choose_arm(self):
        self._sample_theta()
        return self._get_best_arm()


