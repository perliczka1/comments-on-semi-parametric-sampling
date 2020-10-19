import numpy as np


class LinearSemiParametricSampling:
    def __init__(self, arms_nb: int, d: int, X: np.array, sigma_1: float, sigma_2: float, sigma_3: float, seed: int):
        self.seed = seed
        self.generator = np.random.default_rng(self.seed)
        self.N = arms_nb
        self.d = d
        self.X = X  # shape (d, N)
        self.n = np.zeros(self.N)
        self.r_avg = np.zeros(self.N)
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.sigma_3 = sigma_3

        self.A = self._initial_A()
        self.b = self._initial_b()

        self.theta = None  # shape (d, 1)
        self.gamma = None  # shape (N, 1)

    def _initial_A(self) -> np.array:
        return 1 / (self.sigma_3 ** 2) * np.identity(self.d)

    def _initial_b(self) -> np.array:
        return np.zeros((self.d, 1))

    def _delta_A(self) -> np.array:
        delta_A = np.zeros((self.d, self.d))
        for i in range(self.N):
            x = np.expand_dims(self.X[:, i], axis=1)
            delta_A += self.n[i] / (self.sigma_1 ** 2 + self.n[i] * (self.sigma_2 ** 2)) * (x @ x.T)
        return delta_A

    def _delta_b(self) -> np.array:
        delta_b = np.zeros((self.d, 1))
        for i in range(self.N):
            x = np.expand_dims(self.X[:, i], axis=1)
            delta_b += (self.n[i] * self.r_avg[i]) / (self.sigma_1 ** 2 + self.n[i] * (self.sigma_2 ** 2)) * x
        return delta_b

    def _estimate_gamma_mean(self) -> np.array:
        numerator = (self.sigma_2 ** 2) * self.n * self.r_avg + (self.sigma_1 ** 2) * self.theta.T @ self.X
        denominator = self.sigma_1 ** 2 + self.n * (self.sigma_2 ** 2)
        return numerator / denominator

    def _estimate_gamma_var(self):
        return (self.sigma_1 * self.sigma_2) ** 2 / (self.sigma_1 ** 2 + self.n * self.sigma_2 ** 2)

    def _estimate_theta_mean(self) -> np.array:
        A_inv = np.linalg.inv(self.A)
        return A_inv @ self.b

    def _estimate_theta_cov(self) -> np.array:
        A_inv = np.linalg.inv(self.A)
        return A_inv

    def _sample_theta(self) -> None:
        mean = self._estimate_theta_mean()
        mean_flat = mean.flatten()
        cov = self._estimate_theta_cov()
        theta_flat = self.generator.multivariate_normal(mean_flat, cov)
        self.theta = np.expand_dims(theta_flat, axis=1)

    def _sample_gamma(self) -> None:
        mean = self._estimate_gamma_mean()
        mean_flat = mean.flatten()
        gamma_var = self._estimate_gamma_var()
        self.gamma = self.generator.multivariate_normal(mean_flat, np.diag(gamma_var))

    def _update_parameters(self) -> None:
        self.A = self._initial_A() + self._delta_A()
        self.b = self._initial_b() + self._delta_b()
        self.theta = None
        self.gamma = None

    def _update_average(self, current_average: float, current_weight: float, new_value: float) -> float:
        return (current_average * current_weight + new_value) / (current_weight + 1)

    def observe_reward(self, i: int, r: float):
        self.r_avg[i] = self._update_average(self.r_avg[i], self.n[i], r)
        self.n[i] += 1
        self._update_parameters()

    def _get_best_arm(self):
        return np.argmax(self.gamma)

    def choose_arm(self):
        self._sample_theta()
        self._sample_gamma()
        return self._get_best_arm()


