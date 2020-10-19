from collections import namedtuple
import pytest

import numpy as np


@pytest.fixture
def setup():
    from models.linear_gaussian_sampling import LinearGaussianSampling

    X = np.array([[1, 0], [0, 2], [1, 1]])
    model = LinearGaussianSampling(arms_nb=2, d=3, X=X, v=1 / 2, seed=1)

    attributes = {}
    attributes["initial_B"] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    attributes["initial_f"] = np.array([[0], [0], [0]])
    attributes["initial_mi_dashed"] = np.array([[0], [0], [0]])

    observed_reward_kwargs = {'i': 0, 'r': 10}

    attributes["expected_B"] = np.array([[2, 0, 1], [0, 1, 0], [1, 0, 2]])
    attributes["expected_f"] = np.array([[10], [0], [10]])
    expected_B_inv = np.array([[2/3, 0, -1/3], [0, 1, 0], [-1/3, 0, 2/3]])
    attributes["expected_mi_dashed"] = expected_B_inv @ attributes["expected_f"]
    
    attributes["mi_mean"] = attributes["expected_mi_dashed"]
    attributes["mi_cov"] = 1 / 4 * expected_B_inv

    attributes["observed_exp_rewards_mean"] = attributes["mi_mean"].T @ X
    attributes["observed_exp_rewards_cov"] = X.T @ attributes["mi_cov"] @ X
    
    Result = namedtuple('Result', 'model observed_reward_kwargs attributes')

    return Result(model, observed_reward_kwargs, attributes)


def test_init(setup):
    # given
    model = setup.model

    # then
    assert np.allclose(model.B, setup.attributes['initial_B'])
    assert np.allclose(model.f, setup.attributes['initial_f'])
    assert np.allclose(model.mi_dashed, setup.attributes['initial_mi_dashed'])

    assert model.mi is None
    assert model.exp_reward is None


def test_observe_reward(setup):
    # given
    model = setup.model

    # when
    model.observe_reward(**setup.observed_reward_kwargs)

    # then
    assert np.allclose(model.B, setup.attributes['expected_B'])
    assert np.allclose(model.f, setup.attributes['expected_f'])
    assert np.allclose(model.mi_dashed, setup.attributes['expected_mi_dashed'])

    assert model.mi is None
    assert model.exp_reward is None


def test_sample_mi(setup):
    # given
    model = setup.model

    # when
    model.observe_reward(**setup.observed_reward_kwargs)

    steps = 10000
    observed_mis = []
    for t in range(steps):
        model._sample_mi()
        observed_mis.append(model.mi)
    observed_mis_array = np.hstack(observed_mis)
    observed_mis_mean = np.mean(observed_mis_array, axis=1, keepdims=True)
    observed_mis_cov = np.cov(observed_mis_array)

    # then
    assert np.allclose(observed_mis_mean, setup.attributes['mi_mean'], atol=0.1, rtol=0)
    assert np.allclose(observed_mis_cov, setup.attributes['mi_cov'], atol=0.1, rtol=0)


def test_update_exp_reward(setup):
    # given
    model = setup.model

    # when
    model.observe_reward(**setup.observed_reward_kwargs)

    steps = 10000
    observed_exp_rewards = []
    for t in range(steps):
        model._sample_mi()
        model._update_exp_reward()
        observed_exp_rewards.append(model.exp_reward)
    observed_exp_rewards_array = np.vstack(observed_exp_rewards)
    observed_exp_rewards_mean = np.mean(observed_exp_rewards_array, axis=0, keepdims=True)
    observed_exp_rewards_cov = np.cov(observed_exp_rewards_array.T)

    # then
    assert np.allclose(observed_exp_rewards_mean, setup.attributes['observed_exp_rewards_mean'], atol=0.1, rtol=0)
    assert np.allclose(observed_exp_rewards_cov, setup.attributes['observed_exp_rewards_cov'], atol=0.1, rtol=0)


