#!/bin/bash
MODEL_PATH="models"

DATASET="$1"
RUNS=$2

EPOCHS=${3:-1000}
VERSION=${4:-6}

echo "######### Running experiment for $DATASET #########"
for i in $(seq 1 ${RUNS}); do
	model_run_path="${MODEL_PATH}/${DATASET}.v${VERSION}_${i}"
	if [ ! -f "${model_run_path}/model_checkpoint" ]; then
		python3.9 run_experiment.py --model-root="${model_run_path}" --dataset="${DATASET}_clean_${i}" --epochs=${EPOCHS}
	fi
done
