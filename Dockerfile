# Specifing base image and tag
FROM python:3.7.8
WORKDIR /root

# Installing packages
COPY requirements_model.txt /root/requirements_model.txt
RUN pip install -r /root/requirements_model.txt
RUN pip install gsutil

# Copying credentials for Google Cloud
COPY credentials.json credentials.json
RUN chmod 700 credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS "credentials.json"
RUN echo "$GOOGLE_APPLICATION_CREDENTIALS\nsemi-parametric-sampling\nN" | gsutil config -e

# Copying the code to the docker image.
COPY models/ /root/comments-on-semi-parametric-sampling/models
COPY simulation/ /root/comments-on-semi-parametric-sampling/simulation

ENV PYTHONPATH "${PYTHONPATH}:/root/comments-on-semi-parametric-sampling"

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "comments-on-semi-parametric-sampling/simulation/run.py"]





