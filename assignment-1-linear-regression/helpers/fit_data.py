#!/usr/bin/env python3

import sklearn
import sklearn.linear_model
import joblib
import logging

def fit_data(X_train, y_train, model_filename, regressor):
    """Non Scaling Fit Data to Model"""
    logging.info(f"Fitting data to model...")

    # do the fit/training
    regressor.fit(X_train, y_train)

    # save the trained model
    joblib.dump(regressor, model_filename)
    
    logging.info(f"Model fit complete")
    
    return regressor
