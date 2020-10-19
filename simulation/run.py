import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from inspect import getfullargspec
import argparse

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.linear_semi_parametric_sampling import LinearSemiParametricSampling
from models.beta_priors_sampling import BetaPriorsSampling
from models.gaussian_priors_sampling import GaussianPriorsSampling
from models.linear_gaussian_sampling import LinearGaussianSampling
from simulation.environment import Environment
from simulation.log_saver import LogSaver


def run(steps: int, environment_kwargs: Dict, models: Dict[type, Dict], log_saver: LogSaver) -> defaultdict:
    env = Environment(**environment_kwargs)
    log_saver.save_class(env, "environment")
    models_instances = []
    for model_class, model_kwargs in models.items():
        if "X" in model_kwargs.keys():
            model_kwargs["X"] = env.X
        model = model_class(**model_kwargs)
        models_instances.append(model)

    regrets = defaultdict(list)
    arms = defaultdict(list)
    for t in tqdm(range(steps)):
        selected_arms = {}
        observed_rewards = {}
        observed_regrets = {}
        for model in models_instances:
            arm = model.choose_arm()
            selected_arms[model] = arm
            log_saver.save_class_for_step(model, type(model).__name__, t)
        for arm in set(selected_arms.values()):
            observed_rewards[arm] = env.get_reward(arm)
            observed_regrets[arm] = env.get_regret(arm)

        for model in models_instances:
            arm = selected_arms[model]
            regret = observed_regrets[arm]
            model.observe_reward(arm, observed_rewards[arm])
            regrets[type(model).__name__].append(float(regret))
            arms[type(model).__name__].append(int(arm))

    log_saver.save_dict(dict(arms), "arms")
    log_saver.save_dict(dict(regrets), "regrets")
    return regrets


def get_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps")
    parser.add_argument("--models", type=str, required=True, nargs="+", help="Names of the model classes",
                        choices=["LinearSemiParametricSampling", "BetaPriorsSampling",
                                 "GaussianPriorsSampling", "LinearGaussianSampling"])
    parser.add_argument("--arms_nb", type=int, required=True, help="Number of arms")
    parser.add_argument("--a", type=float, required=True, help="Norm of linear parameter vector")
    parser.add_argument("--d", type=int, required=True, help="Dimensionality of linear parameter vector")
    parser.add_argument("--reward_distribution", type=str, choices=["normal", "binomial"], required=True,
                        help="Dimensionality of linear parameter vector")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed used in all randomized aspects of a simulation and model")
    parser.add_argument("--sigma_1", type=float, required=False,
                        help="LSPS hyper-parameter: Standard deviation of reward (r) conditionally on expected reward (gamma)")
    parser.add_argument("--sigma_2", type=float, required=False,
                        help="LSPS hyper-parameter: Standard deviation of expected reward (gamma) conditionally on linear paramter vector (theta)")
    parser.add_argument("--sigma_3", type=float, required=False,
                        help="LSPS hyper-parameter: Standard deviation of linear paramter vector (theta)")
    parser.add_argument("--v", type=float, required=False, help="Linear Gaussian sampling hyper-parameter")
    parser.add_argument("--save_every", type=int, required=True, help="Save every specified number of steps")
    parser.add_argument("--gcs_bucket_path", type=str, required=False,
                        help="Google Cloud Storage path to save the results e.g. gs://semi-parametric-sampling-bucket")
    return parser.parse_args(args)


def get_string_from_current_time() -> str:
    return datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S%f")


def uniquize_experiment_name(name: str) -> str:
    return name + "_" + get_string_from_current_time()


def prepare_models(args: argparse.Namespace) -> Dict[(str, Dict)]:
    models = args.models
    models_prepared = {}
    for m in models:
        model_class = globals()[m]
        model_init_argument_names = getfullargspec(model_class.__init__).args
        model_kwargs = {argument: vars(args).get(argument) for argument in model_init_argument_names if
                        argument != "self"}
        models_prepared[model_class] = model_kwargs
    return models_prepared


def main(args: argparse.Namespace) -> None:
    args = get_args(args)
    log_saver = LogSaver(uniquize_experiment_name(args.name), "./logging", args.gcs_bucket_path,
                         args.save_every)

    environment_kwargs = {"N": args.arms_nb, "a": args.a, "d": args.d, "reward_distribution": args.reward_distribution,
                          "seed": args.seed}

    models_prepared = prepare_models(args)
    log_saver.save_dict(environment_kwargs, "environment_kwargs")
    [log_saver.save_dict(model_kwargs, f"{model.__name__}_kwargs") for model, model_kwargs in models_prepared.items()]

    regrets = run(args.steps, environment_kwargs, models_prepared, log_saver)
    log_saver.sync_with_gcs()


if __name__ == "__main__":
    main(sys.argv[1:])
