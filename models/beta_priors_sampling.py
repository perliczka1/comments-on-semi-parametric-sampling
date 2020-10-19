import numpy as np
import pandas as pd


class BetaPriorsSampling:
    """
    From the paper:
    Near-optimal Regret Bounds for Thompson Sampling
    http://www.columbia.edu/~sa3305/j3.pdf
    """
    def __init__(self, arms_nb: int, seed: int):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.N = arms_nb
        self.S = np.zeros(self.N)
        self.F = np.zeros(self.N)
        self.theta = None # shape N

    def _sample_theta(self) -> None:
        self.theta = self.generator.beta(self.S + 1, self.F + 1)

    def observe_reward(self, i: int, r: float):
        if r == 1:
            self.S[i] += 1
        elif r == 0:
            self.F[i] += 1
        else:
            raise ValueError("Reward can be only 0 or 1 for Thompson sampling with Beta priors.")
        self.theta = None

    def _get_best_arm(self):
        return np.argmax(self.theta)

    def choose_arm(self):
        self._sample_theta()
        return self._get_best_arm()


