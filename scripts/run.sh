#!/bin/bash
MODEL_PATH="models"

DATASET="$1"

EPOCHS=${2:-300}
VERSION=${3:-6}

case ${DATASET} in
	enwik6)
		LOW=5
		HIGH=1000
		RUNS=10
		;;
	enwik7)
		LOW=5
		HIGH=10000
		RUNS=10
		;;
	enwik8)
		LOW=5
		HIGH=100000
		RUNS=1
		;;
	enwik9)
		LOW=5
		HIGH=1000000
		RUNS=1
		;;
	*)
		echo "Invalid dataset" 2>&1
		exit 1
esac

echo "######### Running experiment for $DATASET #########"
for i in $(seq 1 ${RUNS}); do
	model_run_path="${MODEL_PATH}/${DATASET}.v${VERSION}_${i}_min_alpha"
	if [[ ${RUNS} -eq 1 ]]; then
		dataset_path="${DATASET}_clean"
	else
		dataset_path="${DATASET}_clean_${i}"
	fi
	if [ ! -f "${model_run_path}/model_checkpoint" ]; then
		python3.9 run_experiment.py \
				--model-root="${model_run_path}" \
				--dataset="${dataset_path}" \
				--epochs=${EPOCHS} \
				--low=${LOW} \
				--high=${HIGH}
	fi
done
