# Arguments
# $1 - name of the directory with job specification
# $2 - number of nodes in a cluster - to use significant number you might need to increase quotas on number of CPUs here: https://console.cloud.google.com/iam-admin/quotas
# $3 - machine type - recommended: e2-standard-2 for testing; n2-standard-2 for final calculations

gcloud config set account $COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_EMAIL
gcloud container clusters create comments-on-semi-parametric-sampling --region europe-west4 --node-locations europe-west4-a --num-nodes=$2 --machine-type $3 --no-enable-ip-alias --project $COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT
gcloud beta container clusters get-credentials comments-on-semi-parametric-sampling --region europe-west4 --project $COMMENTS_ON_SEMI_PARAMETRIC_SAMPLING_PROJECT
kubectl delete jobs --all
kubectl create -f $1/
kubectl get pods


