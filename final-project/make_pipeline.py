#!/usr/bin/env python3

################################################################
#
# These custom functions help with constructing common pipelines.
# They make use of my_args, and object that has been configured
# by the argparse module to match user requests.
#
from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.tree
import sklearn.neighbors

def make_numerical_predictor_params(my_args):
    params = { 
        "features__numerical__numerical-features-only__do_predictors" : [ True ],
        "features__numerical__numerical-features-only__do_numerical" : [ True ],
    }
    if my_args.numerical_missing_strategy:
        params["features__numerical__missing-data__strategy"] = [ 'median', 'mean', 'most_frequent' ]
    if my_args.use_polynomial_features:
        params["features__numerical__polynomial-features__degree"] = [ 2 ] # [ 1, 2, 3 ]

    return params

def make_categorical_predictor_params(my_args):
    params = { 
        "features__categorical__categorical-features-only__do_predictors" : [ True ],
        "features__categorical__categorical-features-only__do_numerical" : [ False ],
        "features__categorical__encode-category-bits__categories": [ 'auto' ],
        "features__categorical__encode-category-bits__handle_unknown": [ 'ignore' ],
    }
    if my_args.categorical_missing_strategy:
        params["features__categorical__missing-data__strategy"] = [ 'most_frequent' ]
    return params

def make_predictor_params(my_args):
    p1 = make_numerical_predictor_params(my_args)
    p2 = make_categorical_predictor_params(my_args)
    p1.update(p2)
    return p1

def make_SGD_params(my_args):
    raise NotImplementedError("SGD not implemented yet")

def make_linear_params(my_args):
    raise NotImplementedError("Linear not implemented yet")

def make_logistic_params(my_args):
    logistic_params = {
        "model__penalty": [ "l1", "l2", "elasticnet", "none" ],
        "model__dual": [ True, False ],
        "model__tol": [ 0.0001, 0.001, 0.01 ],
        "model__C": [ 0.1, 0.5, 1.0, 2.0, 5.0 ],
        "model__fit_intercept": [ True, False ],
        "model__intercept_scaling": [ 1, 2, 5 ],
        "model__class_weight": [ None ], # [ None, "balanced" ],
        "model__random_state": [ my_args.random_seed ], # [ None, 0, 1, 2, 4, 8 ],
        "model__solver": [ "lbfgs" ], # [ "newton-cg", "lbfgs", "liblinear", "sag", "saga", "auto" ],
        "model__max_iter": [ 1000 ], # [ 100, 1000, 10000 ],
        "model__verbose": [ 0 ], # [ 0, 1, 2, 3 ],
        "model__warm_start": [ False ], # [ True, False ],
        "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ],
    }
    return logistic_params

def make_SVM_params(my_args):
    raise NotImplementedError("SVM not implemented yet")

def make_boost_params(my_args):
    raise NotImplementedError("Boosting not implemented yet")

def make_forest_params(my_args):
    raise NotImplementedError("Random Forest not implemented yet")

def make_tree_params(my_args):
    tree_params = {
        "model__criterion": [ "entropy" ], # [ "entropy", "gini" ],
        "model__splitter": [ "best" ], # [ "best", "random" ],
        "model__max_depth": [ 1, 2, 3, 4, None ],
        "model__min_samples_split": [ 2 ], # [ 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64 ],
        "model__min_samples_leaf":  [ 1 ],  # [ 0.01, 0.02, 0.04, 0.1 ],
        "model__max_features":  [ None ], # [ "sqrt", "log2", None ],
        "model__max_leaf_nodes": [ None ], # [ 2, 4, 8, 16, 32, 64, None ],
        "model__min_impurity_decrease": [ 0.0 ], # [ 0.0, 0.01, 0.02, 0.04, 0.1, 0.2 ],
    }
    return tree_params

def make_knn_params(my_args):
    knn_params = {
        "model__n_neighbors": [ 20 ], # [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ], default=5,
        "model__weights": [ "distance" ], # [ "uniform", "distance" ], default="uniform",
        "model__algorithm": [ "auto" ], # [ "auto", "ball_tree", "kd_tree", "brute" ], default="auto",
        "model__leaf_size": [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ], # [ 10, 20, 30, 40, 50 ], default=30,
        "model__p": [ 1 ], # [ 1, 2 ], default=2,
        "model__metric": [ "minkowski" ], # [ "minkowski", "manhattan", "euclidean", "chebyshev" ], default="minkowski",
        "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ], default=None
    }
    return knn_params

def make_SGDc_params(my_args):
    SGDc_params = {
        "model__loss": [ "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron" ], # [ "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron", "squared_error", "huber", "epsilon_insensitive", squared_epsilon_insensitive ], default="hinge",
        "model__penalty": [ "l2", "l1", "elasticnet" ], # [ "l2", "l1", "elasticnet" ], default="l2",
        "model__alpha": [ 0.0001, 0.001, 0.01, 0.1 ], # [ 0.0001, 0.001, 0.01, 0.1 ], default=0.0001,
        "model__l1_ratio": [ 0.15, 0.25, 0.5, 0.75, 0.9 ], # default=0.15,
        "model__fit_intercept": [ True, False ], # default=True,
        "model__max_iter": [ 1000, 2000, 4000, 8000 ], # default=1000,
        "model__tol": [ 0.001, 0.01, 0.1 ], # default=0.001,
        "model__shuffle": [ True, False ], # default=True,
        "model__epsilon": [ 0.1, 0.2, 0.4, 0.8 ], # default=0.1,
        "model__n_jobs": [ -1 ], # [ None, -1, 1, 2, 4, 8 ], default=None,
        "model__random_state": [ my_args.random_seed ], # [ None, 0, 1, 2, 4, 8 ], default=None,
        "model__learning_rate": [ "optimal", "invscaling", "adaptive", "constant" ], # default="optimal",
        "model__eta0": [ 0.0, 0.01, 0.1 ], # default=0.0
        "model__power_t": [ 0.5, 0.75, 1.0 ], # default=0.5,
        "model__early_stopping": [ False ], # [ True, False ], default=False,
        "model__validation_fraction": [ 0.1, 0.2, 0.4, 0.8 ], # default=0.1,
        "model__n_iter_no_change": [ 5, 10, 20, 40 ], # default=5,
        "model__class_weight": [ None, "balanced" ], # default=None,
        "model__warm_start": [ False ], # [ True, False ], default=False,
        "model__average": [ False ], # [ True, False ], default=False,
    }
    return SGDc_params

def make_fit_params(my_args):
    params = make_predictor_params(my_args)
    if my_args.model_type == "SGD":
        model_params = make_SGD_params(my_args)
    elif my_args.model_type == "linear":
        model_params = make_linear_params(my_args)
    elif my_args.model_type == "SVM":
        model_params = make_SVM_params(my_args)
    elif my_args.model_type == "boost":
        model_params = make_boost_params(my_args)
    elif my_args.model_type == "forest":
        model_params = make_forest_params(my_args)
    elif my_args.model_type == "tree":
        model_params = make_tree_params(my_args)
    elif my_args.model_type == "knn":
        model_params = make_knn_params(my_args)
    elif my_args.model_type == "SGDc":
        model_params = make_SGDc_params(my_args)
    elif my_args.model_type == "logistic":
        model_params = make_logistic_params(my_args)
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest, knn, SGDc]".format(my_args.model_type))

    params.update(model_params)
    return params

def make_numerical_feature_pipeline(my_args):
    items = []

    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))

    if my_args.numerical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.numerical_missing_strategy)))
    if my_args.use_polynomial_features:
        items.append(("polynomial-features", sklearn.preprocessing.PolynomialFeatures(degree=my_args.use_polynomial_features)))
    if my_args.use_scaler:
        items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    items.append(("noop", PipelineNoop()))
    
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline(my_args):
    items = []
    
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))

    if my_args.categorical_missing_strategy:
        items.append(("missing-data", sklearn.impute.SimpleImputer(strategy=my_args.categorical_missing_strategy)))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline(my_args):
    """
    Numerical features and categorical features are usually preprocessed
    differently. We split them out here, preprocess them, then merge
    the preprocessed features into one group again.
    """
    items = []

    items.append(("numerical", make_numerical_feature_pipeline(my_args)))
    items.append(("categorical", make_categorical_feature_pipeline(my_args)))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline


def make_fit_pipeline_regression(my_args):
    """
    These are all regression models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDRegressor(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.LinearRegression()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVR()))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingRegressor()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestRegressor()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeRegressor()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline_classification(my_args):
    """
    These are all classification models.
    """
    items = []
    items.append(("features", make_feature_pipeline(my_args)))
    if my_args.model_type == "SGD":
        items.append(("model", sklearn.linear_model.SGDClassifier(max_iter=10000, n_iter_no_change=100, penalty=None))) # verbose=3, 
    elif my_args.model_type == "linear":
        items.append(("model", sklearn.linear_model.RidgeClassifier()))
    elif my_args.model_type == "SVM":
        items.append(("model", sklearn.svm.SVC(probability=True)))
    elif my_args.model_type == "boost":
        items.append(("model", sklearn.ensemble.GradientBoostingClassifier()))
    elif my_args.model_type == "forest":
        items.append(("model", sklearn.ensemble.RandomForestClassifier()))
    elif my_args.model_type == "tree":
        items.append(("model", sklearn.tree.DecisionTreeClassifier()))
    elif my_args.model_type == "knn":
        items.append(("model", sklearn.neighbors.KNeighborsClassifier()))
    elif my_args.model_type == "SGDc":
        items.append(("model", sklearn.linear_model.SGDClassifier()))
    elif my_args.model_type == "logistic":
        items.append(("model", sklearn.linear_model.LogisticRegression()))
    else:
        raise Exception("Unknown model type: {} [SGD, linear, SVM, boost, forest, knn, SGDc]".format(my_args.model_type))

    return sklearn.pipeline.Pipeline(items)

def make_fit_pipeline(my_args):
    return make_fit_pipeline_classification(my_args)
