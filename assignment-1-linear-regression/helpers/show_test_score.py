#!/usr/bin/env python3

import sklearn
import pandas as pd
import joblib

def show_test_score(glb):
    test_data = pd.read_csv(glb.test_filename, index_col=0)
    X_test = test_data[glb.feature_names]
    y_test = test_data[glb.label_name]

    # load model
    regressor = joblib.load(glb.model_filename)

    if isinstance(regressor, tuple):
        regressor, scaler = regressor
        x_test = test_data[glb.feature_names]
        x_test = scaler.transform(x_test)
    else:
        x_test = test_data[glb.feature_names]
        y_test = test_data[glb.label_name]

    # ask model to score the data
    score_test = regressor.score(X_test, y_test)
    print("Test Score:")
    print("R^2: {}".format(score_test))

    # show mean square error and mean absolute error
    y_predicted = regressor.predict(X_test)
    loss_test = sklearn.metrics.mean_squared_error(y_test, y_predicted)
    print("MSE: {}".format(loss_test))
    loss_test = sklearn.metrics.mean_absolute_error(y_test, y_predicted)
    print("MAE: {}".format(loss_test))
    print()
