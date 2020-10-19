import itertools
import shutil
import os
from copy import deepcopy
from types import ModuleType
from typing import List, Dict, Any
import importlib.util

import yaml
import argparse
import sys


def import_module_from_path(path: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate specification for Kubernetes jobs")
    parser.add_argument("--job_specs_input_path", type=str, required=True,
                        help="Path to python file with defined options for jobs e.g. job_specs_input_hp_tuning.py")
    return parser.parse_args(args)


def generate_possible_models_arguments(hp_tuning_parameters: Dict[str, Dict], available_models: List[str]) -> List[Dict]:
    models_arguments_from_cartesian_product = []
    if hp_tuning_parameters:
        for model, params in hp_tuning_parameters.items():
            if model in available_models:
                params_names = list(params.keys())
                params_values = list(params.values())
                for single_values_tuple in itertools.product(*params_values):
                    arguments = ["--models", model]
                    for i, value in enumerate(single_values_tuple):
                        arguments.append(params_names[i])
                        arguments.append(value)
                    models_arguments_from_cartesian_product.append(arguments)
    return models_arguments_from_cartesian_product


def generate_possible_arguments(module: ModuleType) -> List:
    possible_arguments = []
    for r_d in module.reward_distributions:
        available_models = module.models_for_environment[r_d]
        possible_model_arguments_from_hp_tuning = generate_possible_models_arguments(module.hp_tuning_parameters, available_models)
        for s in module.seeds:
            for a in module.a_values:
                single_arguments_configuration = module.args + ["--reward_distribution", r_d] + ["--seed", s] + \
                                                 ["--a", a] + module.environment_dependent_parameters.get((r_d, a), [])
                if possible_model_arguments_from_hp_tuning:
                    for model_args in possible_model_arguments_from_hp_tuning:
                        single_arguments_configuration_with_model_args = single_arguments_configuration + model_args
                        possible_arguments.append(single_arguments_configuration_with_model_args)
                else:
                    single_arguments_configuration += ["--models"] + available_models
                    possible_arguments.append(single_arguments_configuration)
    return possible_arguments


def combine_kubernetes_params_and_run_agrs(kubernetes_params: Dict[str, Any], args: List[str],
                                           **name_format_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    all_params = deepcopy(kubernetes_params)
    all_params["metadata"]["name"] = all_params["metadata"]["name"].format(**name_format_kwargs)
    all_params["spec"]["template"]["spec"]["containers"][0]["args"] = args
    return all_params


def main(args):
    args = get_args(args)
    specs_module = import_module_from_path(args.job_specs_input_path, "specs")
    possible_arguments = generate_possible_arguments(specs_module)

    shutil.rmtree(specs_module.job_specs_path, ignore_errors=True)
    os.makedirs(specs_module.job_specs_path)

    for i, args in enumerate(possible_arguments):
        additional_args = ["--name", specs_module.name_format_string.format(i=i)]
        complete_job_specs = combine_kubernetes_params_and_run_agrs(specs_module.kubernetes, additional_args + args,
                                                                    i=i)
        file_path = os.path.join(specs_module.job_specs_path, f"job_{i}.yaml")
        with open(file_path, "w+") as f:
            yaml.dump(complete_job_specs, f)


if __name__ == "__main__":
    main(sys.argv[1:])
