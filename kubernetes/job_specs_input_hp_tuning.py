import os

kubernetes = {"apiVersion": "batch/v1",
              "kind": "Job",
              "metadata": {"name": "comments-on-semi-parametric-sampling-job-{i}"},
              "spec": {
                  "template": {"metadata": {"labels": {"app": "comments-on-semi-parametric-sampling-job"},
                                            "name": "comments-on-semi-parametric-sampling-job"},
                               "spec": {
                                   "restartPolicy": "Never",
                                   "containers": [
                                       {"image": f"gcr.io/{os.environ['COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT']}/semi-parametric-sampling:latest",
                                        "name": "semi-parametric-sampling",
                                        "resources": {
                                            "requests":
                                                {"memory": "4000Mi"}}}]}}}}

args = ["--steps", "25000", "--save_every", "1000", "--arms_nb", "1000", "--d", "5", "--gcs_bucket_path",
        f"gs://{os.environ['COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_BUCKET']}"]

hp_tuning_parameters = {"LinearSemiParametricSampling": {"--sigma_1": ["0.1", "1", "10"],
                                                         "--sigma_2": ["0.1", "1", "10"],
                                                         "--sigma_3": ["0.1", "1", "10"]},
                        "LinearGaussianSampling": {"--v": ["0.1", "1", "10"]},
                        "GaussianPriorsSampling": {},
                        "BetaPriorsSampling": {}}

environment_dependent_parameters = {}

models_for_environment = {
    "binomial": ["LinearSemiParametricSampling", "LinearGaussianSampling", "GaussianPriorsSampling",
                 "BetaPriorsSampling"],
    "normal": ["LinearSemiParametricSampling", "LinearGaussianSampling", "GaussianPriorsSampling"]}

reward_distributions = ["binomial", "normal"]

seeds = ["1", "2", "3"]

a_values = ["0.5", "1.0"]

name_format_string = "hp_tuning_{i}"

job_specs_path = "job_specs_hp_tuning"
