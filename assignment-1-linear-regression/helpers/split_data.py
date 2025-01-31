#!/usr/bin/env python3

import sklearn
import logging

def split_data(glb):
    
    logging.info(f"Splitting data...")
    logging.info(f"train_filename: {glb.train_filename}")
    logging.info(f"test_filename: {glb.test_filename}")
    logging.info(f"seed: {glb.seed}")
    logging.info(f"ratio: {glb.ratio}")
    data_train, data_test = \
        sklearn.model_selection.train_test_split(glb.data, test_size=glb.ratio, random_state=glb.seed)

    logging.info(f"Saving data...")
    data_train.to_csv(glb.train_filename)
    data_test.to_csv(glb.test_filename)
    logging.info(f"Data Split Complete")
    
    return data_train, data_test
