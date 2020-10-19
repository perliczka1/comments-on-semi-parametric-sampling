import numpy as np


class Environment:
    def __init__(self, N: int, a: int, d: int, reward_distribution: str, seed: int):
        self.N = N
        self.a = a
        self.d = d
        self.seed = seed
        self.generator = np.random.default_rng(seed=self.seed)

        self.reward_distribution = reward_distribution
        self.X = self.generator.normal(size=(d, N))
        self.X = self._prepare_context_features(self.X)

        self.theta = self.generator.normal(size=(d, 1))
        self.theta = self._prepare_linear_parameter_vector(self.theta, self.a)

        self.bias = self.generator.uniform(low=0, high=1-self.a, size=(1, N))

        self.expected_rewards = self._prepare_expected_rewards(self.theta, self.X, self.bias)

    @staticmethod
    def _prepare_context_features(X: np.array) -> np.array:
        result_X = np.abs(X)
        max_norm = np.linalg.norm(result_X, ord=2, axis=0).max()
        result_X = result_X / max_norm
        return result_X

    @staticmethod
    def _prepare_linear_parameter_vector(theta: np.array, a: float) -> np.array:
        result_theta = np.abs(theta)
        norm = np.linalg.norm(result_theta, ord=2)
        result_theta = result_theta / norm * a
        return result_theta

    @staticmethod
    def _prepare_expected_rewards(theta: np.array, X: np.array, bias: np.array) -> np.array:
        return theta.T @ X + bias

    def get_reward(self, arm: int) -> float:
        expected_value = self.expected_rewards[0, arm]
        if self.reward_distribution == "normal":
            return self.generator.normal(loc=expected_value)
        elif self.reward_distribution == "binomial":
            return self.generator.binomial(n=1, p=expected_value)

    def get_regret(self, arm: int) -> float:
        expected_reward = self.expected_rewards[0, arm]
        optimal_expected_reward = self.expected_rewards.max()
        return optimal_expected_reward - expected_reward



