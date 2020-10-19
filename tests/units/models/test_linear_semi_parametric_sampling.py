from collections import namedtuple
import pytest

import numpy as np


@pytest.fixture
def lsps_setup():
    from models.linear_semi_parametric_sampling import LinearSemiParametricSampling

    X = np.array([[1, 0], [0, 2], [1, 1]])
    lsps = LinearSemiParametricSampling(arms_nb=2, d=3, X=X, sigma_1=1 / 2, sigma_2=1 / 3, sigma_3=2, seed=1)

    attributes = {}
    attributes["initial_A"] = 0.25 * np.identity(3)
    attributes["initial_b"] = np.array([[0], [0], [0]])

    observe_reward_kwargs = {'i': 0, 'r': 10}
    attributes["r_avg"] = np.array([10, 0])
    attributes["n"] = np.array([1, 0])

    attributes["delta_A"] = 36 / 13 * np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    attributes["delta_b"] = 10 / (13 / 36) * np.array([[1], [0], [1]])
    attributes["theta"] = np.array([0.5, 1, -1])
    attributes["gamma_mean"] = np.array([[71 / 26, 1]])
    attributes["gamma_var"] = np.array([1 / 13, 1 / 9])

    attributes["expected_A"] = attributes["initial_A"] + attributes["delta_A"]
    attributes["expected_b"] = attributes["initial_b"] + attributes["delta_b"]

    expected_A_inv = [[628 / 301, 0, -576 / 301], [0, 4, 0], [-576 / 301, 0, 628 / 301]]
    attributes["theta_mean"] = expected_A_inv @ attributes["expected_b"]
    attributes["theta_cov"] = expected_A_inv

    Result = namedtuple('Result', 'lsps observe_reward_kwargs attributes')

    return Result(lsps, observe_reward_kwargs, attributes)


def test_init(lsps_setup):
    # when
    lsps = lsps_setup.lsps

    # then
    assert np.allclose(lsps.A, lsps_setup.attributes['initial_A'])
    assert np.allclose(lsps.b, lsps_setup.attributes['initial_b'])
    assert lsps.n.sum() == 0
    assert lsps.r_avg.sum() == 0


def test_observe_reward(lsps_setup):
    # given
    lsps = lsps_setup.lsps

    # when
    lsps_setup.lsps.observe_reward(**lsps_setup.observe_reward_kwargs)

    # then
    assert np.allclose(lsps.r_avg, lsps_setup.attributes['r_avg'])
    assert np.allclose(lsps.n, lsps_setup.attributes['n'])
    assert np.allclose(lsps.A, lsps_setup.attributes['expected_A'])
    assert np.allclose(lsps.b, lsps_setup.attributes['expected_b'])
    assert lsps.theta is None
    assert lsps.gamma is None


def test_sample_theta(lsps_setup):
    # given
    lsps = lsps_setup.lsps

    # when
    lsps.observe_reward(**lsps_setup.observe_reward_kwargs)

    steps = 10000
    observed_thetas = []
    for t in range(steps):
        lsps._sample_theta()
        observed_thetas.append(lsps.theta)
    observed_thetas_array = np.hstack(observed_thetas)
    observed_theta_mean = np.mean(observed_thetas_array, axis=1, keepdims=True)
    observed_cov = np.cov(observed_thetas_array)

    # then
    assert np.allclose(observed_theta_mean, lsps_setup.attributes['theta_mean'], atol=0.1, rtol=0)
    assert np.allclose(observed_cov, lsps_setup.attributes['theta_cov'], atol=0.1, rtol=0)


def test_estimate_gamma_mean(lsps_setup):
    # given
    lsps = lsps_setup.lsps

    # when
    lsps.observe_reward(**lsps_setup.observe_reward_kwargs)
    lsps.theta = lsps_setup.attributes["theta"]
    gamma_mean = lsps._estimate_gamma_mean()

    # then
    assert np.allclose(gamma_mean, lsps_setup.attributes["gamma_mean"])


def test_estimate_gamma_var(lsps_setup):
    # given
    lsps = lsps_setup.lsps

    # when
    lsps.observe_reward(**lsps_setup.observe_reward_kwargs)
    lsps.theta = lsps_setup.attributes["theta"]
    gamma_var = lsps._estimate_gamma_var()

    # then
    assert np.allclose(gamma_var, lsps_setup.attributes["gamma_var"])


def test_sample_gamma(lsps_setup):
    # given
    lsps = lsps_setup.lsps

    # when
    lsps.observe_reward(**lsps_setup.observe_reward_kwargs)
    lsps.theta = lsps_setup.attributes["theta"]

    steps = 1000
    observed_gammas = []
    for t in range(steps):
        lsps._sample_gamma()
        observed_gammas.append(lsps.gamma)
    observed_gammas_array = np.vstack(observed_gammas)
    observed_gammas_mean = np.mean(observed_gammas_array, axis=0, keepdims=True)
    observed_var = np.var(observed_gammas_array, axis=0, keepdims=True)

    # then
    assert np.allclose(observed_gammas_mean, lsps_setup.attributes['gamma_mean'], atol=0.1, rtol=0)
    assert np.allclose(observed_var, lsps_setup.attributes['gamma_var'], atol=0.1, rtol=0)


