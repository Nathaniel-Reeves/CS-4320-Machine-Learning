#!/bin/bash

if [ ! -e fashion-mnist-mini-train-validate.csv ]; then
    split=../030-regression-fit/split_data.py
    $split --data-file fashion-mnist-mini.csv --test-ratio 0.20
    $split --data-file fashion-mnist-mini-train.csv --test-ratio 0.30 --train-file fashion-mnist-mini-train-fit.csv --test-file fashion-mnist-mini-train-validate.csv
fi

#
# Configure the network architecture here.
#

# Name the model file after the architecture, presumably with some bookkeeping on your part.
model_file="model-description-here.joblib"

# Fit the new model
./cnn_classification.py cnn-fit --train-file fashion-mnist-mini-train-fit.csv --model-file ${model_file}
# Score the new model
./cnn_classification.py score --train-file fashion-mnist-mini-train-fit.csv --test-file fashion-mnist-mini-train-validate.csv --show-test 1 --model-file ${model_file}

#
# Record results, and observations.
# Go to Configure the network architecture above.
#
