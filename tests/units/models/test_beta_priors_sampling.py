from collections import namedtuple
import pytest

import numpy as np


@pytest.fixture
def setup():
    from models.beta_priors_sampling import BetaPriorsSampling

    model = BetaPriorsSampling(arms_nb=2, seed=1)

    attributes = {"initial_S": np.array([0, 0]), "initial_F": np.array([0, 0])}

    observed_reward_kwargs = {'i': 0, 'r': 1}

    attributes["expected_S"] = np.array([1, 0])
    attributes["expected_F"] = np.array([0, 0])

    attributes["theta_mean"] = np.array([2/3, 1/2])
    attributes["theta_var"] = [2/36, 1/12]

    Result = namedtuple('Result', 'model observed_reward_kwargs attributes')

    return Result(model, observed_reward_kwargs, attributes)


def test_init(setup):
    # when
    model = setup.model

    # then
    assert np.allclose(model.S, setup.attributes['initial_S'])
    assert np.allclose(model.F, setup.attributes['initial_F'])
    assert model.theta is None


def test_observe_reward(setup):
    # given
    model = setup.model

    # when
    setup.model.observe_reward(**setup.observed_reward_kwargs)

    # then
    assert np.allclose(model.S, setup.attributes['expected_S'])
    assert np.allclose(model.F, setup.attributes['expected_F'])
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
