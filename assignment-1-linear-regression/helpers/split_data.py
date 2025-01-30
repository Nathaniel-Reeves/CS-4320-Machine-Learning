#!/usr/bin/env python3

import sklearn
import logging

def split_data(data, train_filename, test_filename, seed=42, ratio=0.2):
    
    logging.info(f"Splitting data...")
    logging.info(f"train_filename: {train_filename}")
    logging.info(f"test_filename: {test_filename}")
    logging.info(f"seed: {seed}")
    logging.info(f"ratio: {ratio}")
    data_train, data_test = \
        sklearn.model_selection.train_test_split(data, test_size=ratio, random_state=seed)

    logging.info(f"Saving data...")
    data_train.to_csv(train_filename)
    data_test.to_csv(test_filename)
    logging.info(f"Data Split Complete")
