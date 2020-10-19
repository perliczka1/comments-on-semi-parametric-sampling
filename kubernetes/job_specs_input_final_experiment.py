import os

kubernetes = {'apiVersion': 'batch/v1',
              'kind': 'Job',
              'metadata': {'name': 'comments-on-semi-parametric-sampling-job-{i}'},
              'spec': {
                  'template': {'metadata': {'labels': {'app': 'comments-on-semi-parametric-sampling-job'},
                                            'name': 'comments-on-semi-parametric-sampling-job'},
                               'spec': {
                                   'restartPolicy': 'Never',
                                   'containers': [
                                       {
                                           'image': f"gcr.io/{os.environ['COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT']}/semi-parametric-sampling:latest",
                                           'name': 'semi-parametric-sampling',
                                           'resources': {
                                               'requests':
                                                   {'memory': '5000Mi'}}}]}}}}

args = ["--steps", "200000", "--save_every", "50000", "--arms_nb", "1000", "--d", "5",
        "--gcs_bucket_path", f"gs://{os.environ['COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_BUCKET']}"]

hp_tuning_parameters = {}

environment_dependent_parameters = {
    ("binomial", "0.5"): ["--sigma_1", "0.1", "--sigma_2", "0.1", "--sigma_3", "0.1", "--v", "1.0"],
    ("binomial", "1.0"): ["--sigma_1", "0.1", "--sigma_2", "0.1", "--sigma_3", "1.0", "--v", "1.0"],
    ("normal", "0.5"): ["--sigma_1", "1.0", "--sigma_2", "0.1", "--sigma_3", "0.1", "--v", "1.0"],
    ("normal", "1.0"): ["--sigma_1", "1.0", "--sigma_2", "0.1", "--sigma_3", "0.1", "--v", "1.0"]
}

reward_distributions = ["binomial", "normal"]

models_for_environment = {
    "binomial": ["LinearSemiParametricSampling", "LinearGaussianSampling", "GaussianPriorsSampling",
                 "BetaPriorsSampling"],
    "normal": ["LinearSemiParametricSampling", "LinearGaussianSampling", "GaussianPriorsSampling"]}

seeds = ["4", "5", "6"]

a_values = ["0.5", "1.0"]

name_format_string = "final_experiment_{i}"

job_specs_path = "job_specs_final_experiment"
