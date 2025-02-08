#!/usr/bin/env python3

import sklearn
import joblib



def show_train_score(regressor, X_train, y_train):
    # ask model to score the data
    score_train = regressor.score(X_train, y_train)
    print("Train Score:")
    print("R^2: {}".format(score_train))

    # show mean square error and mean absolute error
    y_predicted = regressor.predict(X_train)
    loss_train = sklearn.metrics.mean_squared_error(y_train, y_predicted)
    print("MSE: {}".format(loss_train))
    loss_train = sklearn.metrics.mean_absolute_error(y_train, y_predicted)
    print("MAE: {}".format(loss_train))
    print()
