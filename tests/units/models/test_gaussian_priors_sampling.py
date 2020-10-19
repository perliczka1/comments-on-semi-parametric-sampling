from collections import namedtuple
import pytest

import numpy as np


@pytest.fixture
def setup():
    from models.gaussian_priors_sampling import GaussianPriorsSampling

    model = GaussianPriorsSampling(arms_nb=2, seed=1)

    attributes = {"initial_k": np.array([0, 0]), "initial_mi": np.array([0, 0])}

    observed_reward_kwargs = {'i': 0, 'r': 10}

    attributes["expected_k"] = np.array([1, 0])
    attributes["expected_mi"] = np.array([5, 0])

    attributes["theta_mean"] = np.array([5, 0])
    attributes["theta_var"] = [0.5, 1]

    Result = namedtuple('Result', 'model observed_reward_kwargs attributes')

    return Result(model, observed_reward_kwargs, attributes)


def test_init(setup):
    # when
    model = setup.model

    # then
    assert np.allclose(model.k, setup.attributes['initial_k'])
    assert np.allclose(model.mi, setup.attributes['initial_mi'])
    assert model.theta is None


def test_observe_reward(setup):
    # given
    model = setup.model

    # when
    setup.model.observe_reward(**setup.observed_reward_kwargs)

    # then
    assert np.allclose(model.k, setup.attributes['expected_k'])
    assert np.allclose(model.mi, setup.attributes['expected_mi'])
    assert model.theta is None


def test_sample_theta(setup):
    # given
    model = setup.model

    # when
    model.observe_reward(**setup.observed_reward_kwargs)

    steps = 10000
    observed_thetas = []
    for t in range(steps):
        model._sample_theta()
        observed_thetas.append(model.theta)
    observed_thetas_array = np.vstack(observed_thetas)
    observed_theta_mean = np.mean(observed_thetas_array, axis=0)
    observed_var = np.var(observed_thetas_array, axis=0)

    # then
    assert np.allclose(observed_theta_mean, setup.attributes['theta_mean'], atol=0.1, rtol=0)
    assert np.allclose(observed_var, setup.attributes['theta_var'], atol=0.1, rtol=0)
