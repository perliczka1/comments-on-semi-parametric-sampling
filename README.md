
# Introduction
This code implements a model presented in [Semi-Parametric Sampling for Stochastic Bandits with Many Arms](https://www.aaai.org/ojs/index.php/AAAI/article/view/4793) and compares it 
to other multi-armed bandit algorithms.
It contains also tuning of hyperparameters. All computation are done on Google Cloud.

In data directory located [here](https://drive.google.com/drive/folders/1GmgGhNnlBVyVPqLRo0g18W9ZTHf15HuW?usp=sharing) on Google Drive there are results from hyperparameter tuning and the final simulation. 
They are described in the document `Comments on "Semi-Parametric Sampling for Stochastic Bandits with Many Arms"`.

# How to reproduce the results?
## Locally 
You can run a single simulation locally. It should finish in less that 20 hours for 200 000 steps, 
depending on your computer.
To do this:
1. Follow instructions from section **3. Create Python virtual environment**. 
2. Run (you may want to change the arguments):
``` bash
source .venv/bin/activate
python simulation/run.py --help # to find out about possible arguments
python simulation/run.py  --name local_testing --steps 1000 --save_every 100 \
                          --arms_nb 10 --a 0.5  --d 100 --models BetaPriorsSampling \
                          --reward_distribution binomial --seed 1
```

## On Google Cloud

#### Disclaimer

It is possible that to succesfully run all the code here you must enable a few services and change quotas on Google Cloud. 
You will get information and detailed instruction on what to do when running scripts provided below.

### 1. Prepare everything on Google Cloud
1. Create an account on [Google Cloud Platform](https://cloud.google.com/). Remember email used for this. 
1. Enable billing (to pay for the computation) or use the Free Trial option.
1. Create a new project. Remember its id. 
1. Create a [bucket](https://cloud.google.com/storage/docs/creating-buckets) to store the data from the experiments. Remember its name.
1. Install [Cloud SDK](https://cloud.google.com/sdk/install) to interact with Google Cloud using command line.
1. Save remembered names as environmental variables – add the following lines
 to your .bash_profile or .zshrc file and reload it.
    ``` bash
    export COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_EMAIL=example@gmail.com
    export COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT=example-project-name
    export COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_BUCKET=example-bucket-name
    ```
1. Authenticate to Google Cloud and set your project as the default one:
    ``` bash
   gcloud auth login
   gcloud config set project $COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT
   ```
1. Create a [service account](https://cloud.google.com/iam/docs/creating-managing-service-accounts#iam-service-accounts-create-console),
then [generate](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#creating_service_account_keys) a JSON key for it. Save it in this project in the main directory as `credentials.json`.
1. [Allow](https://cloud.google.com/storage/docs/access-control/using-iam-permissions) the service account to access the bucket created earlier. 

### 2. Get the project code
 We later assume that all commands are ran from the main project directory.

### 3. Create Python virtual environment
We recommend using Python 3.7.8. 
Please install:
 * [pip](https://pip.pypa.io/en/stable/installing/) 
 * [virtualenv](https://virtualenv.pypa.io/en/latest/).

Then run:
``` bash
virtualenv -p python3.7.8 .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_model.txt
```

### 4. Run tests
``` bash
python -m pytest 
```

### 5. Build and push docker image
Install [docker](https://docs.docker.com/get-docker/) and run:
``` bash
./build_docker.sh
```
It is possible that first you must enable some services on Google Cloud. You will get information and instruction about 
it when running this script.

### 6. Run calculations in the cloud
The code required to run the calculation on the cluster is in `kubernetes` directory.

1. Add a configuration of new calculation by creating new file e.g. `job_specs_input_my_run.py`. 
Instead of *my_run* you can use any other name.
There are provided `job_specs_input_hp_tuning.py` and `job_specs_input_final_experiment.py`
files, used to run the calculations for the publication. 
1. Use generate_jobs_specs.py to generate configuration of all jobs for the cluster:
    ``` bash
    source .venv/bin/activate 
    cd kubernetes
    python generate_jobs_specs.py --job_specs_input_path job_specs_input_my_run.py
    ```
    It will be saved in a place specified by you in the file `job_specs_input_my_run.py`.
1. Assuming that it is `job_specs_my_run` directory we can start the calculations on the cluster by running:
    ``` bash
    cd kubernetes
    ./run_on_cluster.sh job_specs_my_run 10 e2-standard-2 
    ```
    10 is the number of nodes in the cluster, e2-standard-2 is a [type of machine](https://cloud.google.com/compute/docs/machine-types).
    You can change these arguments. The results are going to be saved on Google Cloud Storage in the bucket created in point 1.4.


### 7. Looking at the results 
You can do it in Jupyter Lab. To prepare it please run:
``` bash
source .venv/bin/activate 
pip install jupyterlab "ipywidgets>=7.5"
jupyter labextension install jupyterlab-plotly@4.9.0
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget@4.9.0
python -m ipykernel install --user --name semi_parametric_samping_venv --display-name "Semi-parametric sampling (.venv)"
jupyter lab
```

Examples of the notebooks are provided in `notebooks` directory. 