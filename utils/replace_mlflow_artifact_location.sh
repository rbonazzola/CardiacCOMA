OPATHS=("\/usr\/not-backed-up\/scrb\/src" "\/app" "\/home\/user\/s3" )
NEW_PATH="\/home\/rodrigo\/CISTIB\/repos"
MLFLOW_URI="../mlruns"
EXPERIMENT_PATTERN="\/[0-9]\/"

for ORIGINAL_PATH in ${OPATHS[@]}; do
for EXPERIMENT in `ls ${MLFLOW_URI}`; do 
  for RUN in `ls ${MLFLOW_URI}/$EXPERIMENT/`; do 
    sed -i 's/'${ORIGINAL_PATH}'/'${NEW_PATH}'/' ${MLFLOW_URI}/${EXPERIMENT}/${RUN}/meta.yaml | grep artifact 
    sed -i "s/experiment_id: '.'/experiment_id: '"${EXPERIMENT}"'/g" ${MLFLOW_URI}/${EXPERIMENT}/${RUN}/meta.yaml | grep experiment 
    sed -i "s/"${EXPERIMENT_PATTERN}"/\/"${EXPERIMENT}"\//g" ${MLFLOW_URI}/${EXPERIMENT}/${RUN}/meta.yaml | grep experiment 
    echo $EXPERIMENT/$RUN
  done
done  
done
