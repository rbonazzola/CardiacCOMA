ORIGINAL_PATH="\/usr\/not-backed-up\/scrb\/src"
#ORIGINAL_PATH="\/app"
NEW_PATH="\/home\/rodrigo\/CISTIB\/repos"
MLFLOW_URI="../mlruns"


for EXPERIMENT in `ls ${MLFLOW_URI}`; do 
  for RUN in `ls ${MLFLOW_URI}/$EXPERIMENT/`; do 
    sed -i 's/'${ORIGINAL_PATH}'/'${NEW_PATH}'/' ${MLFLOW_URI}/${EXPERIMENT}/${RUN}/meta.yaml | grep artifact 
    echo $EXPERIMENT/$RUN
  done
done  
