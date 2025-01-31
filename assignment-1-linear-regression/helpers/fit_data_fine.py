#!/usr/bin/env python3

import sklearn
import sklearn.preprocessing
import sklearn.linear_model
import joblib
import logging


def fit_data_fine(X_train, y_train, model_filename, feature_names):

    logging.info(f"Fitting data to model...")

    # scale data with x' = (x - u) / s
    scaler = sklearn.preprocessing.StandardScaler()
    logging.info("Scaling data strategy: StandardScaler")
    # find u and s
    scaler.fit(X_train) 
    # transform data
    X_train = scaler.transform(X_train) 

    # peek at scaled data
    logging.info("Scaled Features")
    logging.info(feature_names)
    logging.info(X_train[:5,:])

    # do the fit/training
    regressor = sklearn.linear_model.SGDRegressor(max_iter=10000)
    regressor.fit(X_train, y_train)

    # save the trained model
    joblib.dump((regressor,scaler), model_filename)

    logging.info(f"Model fit complete")
