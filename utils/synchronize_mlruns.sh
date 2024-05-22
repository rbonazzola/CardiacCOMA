#!/bin/bash

ORIGINAL_EXPID=${1-3}
DESTINY_EXPID=${2:-1}

aws s3 sync --exclude="*pth" s3://users-rbonazzola/CardiacCOMA/mlruns/${ORIGINAL_EXPID}/ ~/01_repos/000_CardiacCOMA/mlruns/${DESTINY_EXPID}
cd ~/01_repos/000_CardiacCOMA/utils
bash replace_mlflow_artifact_location.sh
cd -
