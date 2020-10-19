gcloud config set account $COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_EMAIL

export PROJECT_ID=$COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT
export IMAGE_REPO_NAME=semi-parametric-sampling
export IMAGE_TAG=latest
export IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

# build image
docker build  --no-cache -f Dockerfile -t $IMAGE_URI ./
# check if it works
docker run $IMAGE_URI --name local_testing --steps 10 --save_every 2 --arms_nb 10 --a 0.5  --d 100 --models BetaPriorsSampling --reward_distribution binomial --seed 1 --gcs_bucket_path GS://$COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_BUCKET
# push the image
docker push $IMAGE_URI
