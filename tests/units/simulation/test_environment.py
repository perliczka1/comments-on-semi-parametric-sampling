import numpy as np
from simulation.environment import Environment


def test_prepare_context_features():
    # given
    X = np.array([[1, 1], [0, 1], [-2, -1]]) # d x N matrix

    # when
    result = Environment._prepare_context_features(X)
    result_expected = np.array([[1, 1], [0, 1], [2, 1]]) * (1 / 5 ** 0.5)

    #then
    assert np.allclose(result, result_expected)


def test_prepare_linear_parameter_vector():
    # given
    theta = np.array([[0], [-1], [1]])
    a = 0.5

    # when
    result = Environment._prepare_linear_parameter_vector(theta, a)
    expected_result = np.array([[0], [1], [1]]) * (1 / 2 ** 1.5)

    # then
    assert np.allclose(result, expected_result)


def test_prepare_expected_rewards_normal():
    # given
    X = np.array([[1, 0], [0.1, 0.1], [0.25, 0.3]])
    theta = np.array([[0], [-1], [1]])
    bias = np.array([1, -1])

    # when
    result = Environment._prepare_expected_rewards(theta, X, bias)
    result_expected = np.array([1.15, -0.8])

    # then
    assert np.allclose(result, result_expected)


def test_get_reward_normal():
    # given
    N = 5
    env = Environment(N=N, a=0.9, d=3, reward_distribution="normal", seed=1)

    # when
    steps = 10000
    average_observed_rewards = [0] * N
    for t in range(steps):
        for i in range(N):
            average_observed_rewards[i] += env.get_reward(i) / steps

    # then
    assert np.allclose(env.expected_rewards, np.array([average_observed_rewards]), atol=0.1, rtol=0)


def test_get_reward_binomial():
    # given
    N = 5
    env = Environment(N=N, a=0.9, d=3, reward_distribution="binomial", seed=1)

    # when
    steps = 10000
    average_observed_rewards = [0] * N
    for t in range(steps):
        for i in range(N):
            reward = env.get_reward(i)
            assert reward in (0, 1)
            average_observed_rewards[i] += reward / steps

    # then
    assert np.allclose(env.expected_rewards, np.array([average_observed_rewards]), atol=0.1, rtol=0)