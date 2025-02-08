import sys
import argparse
import logging
import pandas as pd
import sklearn.linear_model
import sklearn.neural_network
import sklearn.ensemble
import joblib

from globals import get_globals
from helpers.explore_data import histogram_all, scatter_all
from helpers.split_data import split_data
from helpers.show_model import show_model, show_function
from helpers.show_model_fine import show_fine_model, show_fine_function
from helpers.show_train_score import show_train_score
from helpers.show_test_score import show_test_score
from helpers.explore_data_fit import scatter_all_with_fit
from helpers.fit_data_fine import fit_data_fine

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Linear Regression')
    parser.add_argument(
        "--logging-level",
        "-l",
        type=str,
        help="logging level: warn, info, debug default: info",
        choices=("warn", "info", "debug"),
        default="info",
    )
    parser.add_argument(
        "--orientation",
        "-o",
        type=str,
        help="orientation: portrait, landscape",
        choices=("portrait", "landscape"),
        default="landscape"
    )
    parser.add_argument(
        "--explore",
        "-e",
        help="explore: histogram, scatter",
        action="store_true"
    )
    parser.add_argument(
        "--prepare",
        "-p",
        help="prepare: clean, transform",
        action="store_true"
    )
    parser.add_argument(
        "--fit",
        "-f",
        help="fit: model",
        action="store_true"
    )
    parser.add_argument(
        "--fit-fine",
        "-sc",
        help="scale: data",
        action="store_true"
    )
    parser.add_argument(
        "--show-model",
        "-s-model",
        help="show: model, function",
        action="store_true"
    )
    parser.add_argument(
        "--show-fine-model",
        "-s-fine-model",
        help="show: model, function",
        action="store_true"
    )
    parser.add_argument(
        "--show-train-score",
        "-s-train",
        help="show: train score",
        action="store_true"
    )
    parser.add_argument(
        "--show-test-score",
        "-s-test",
        help="show: test score",
        action="store_true"
    )
    parser.add_argument(
        "--predict",
        "-predict",
        help="predict: test data (default: data/data-test.csv)",
        type=str,
        default="data/data-test.csv"
    )
    parser.add_argument(
        "--all-steps",
        "-a",
        help="all steps: explore > prepare > fit > show",
        action="store_true"
    )
    parser.add_argument(
        "--all-fine-steps",
        "-af",
        help="all fine steps: explore > prepare > scale > fit > show",
        action="store_true"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="model: class name (default: SGDRegressor)",
        choices=(
            "SGDRegressor",
            "Ridge",
            "Lasso",
            "BayesianRidge",
            "MLPClassifier",
            "LinearRegression",
            "RandomForestRegressor"
        ),
        default="SGDRegressor"
    )
    parser.add_argument(
        "--verbose-level",
        "-v",
        type=int,
        help="verbose: 0, 1, 2",
        choices=(0, 1, 2),
        default=0
    )

    my_args = parser.parse_args(argv[1:])
    if my_args.logging_level == "warn":
        my_args.logging_level = logging.WARN
    elif my_args.logging_level == "info":
        my_args.logging_level = logging.INFO
    elif my_args.logging_level == "debug":
        my_args.logging_level = logging.DEBUG
    else:
        raise ValueError(f"Unknown logging level: {my_args.logging_level}")
    
    if my_args.model == "SGDRegressor":
        my_args.model = sklearn.linear_model.SGDRegressor(
            verbose=my_args.verbose_level,
            max_iter=10000
        )
    elif my_args.model == "Ridge":
        my_args.model = sklearn.linear_model.RidgeCV()
    elif my_args.model == "Lasso":
        my_args.model = sklearn.linear_model.Lasso()
    elif my_args.model == "BayesianRidge":
        my_args.model = sklearn.linear_model.BayesianRidge()
    elif my_args.model == "MLPClassifier":
        my_args.model = sklearn.neural_network.MLPRegressor()
    elif my_args.model == "LinearRegression":
        my_args.model = sklearn.linear_model.LinearRegression(
            n_jobs=-1
        )
    elif my_args.model == "RandomForestRegressor":
        my_args.model = sklearn.ensemble.RandomForestRegressor(
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {my_args.model}")

    return my_args

def main(argv):
    # parse command line arguments, Set logging level
    args = parse_args(argv)
    logging.basicConfig(level=args.logging_level)
    glb = get_globals()
    glb.initialize()

    # first few rows of the DataFrame
    logging.debug(f"data.csv HEAD\n {glb.data.head()}")
    
    
    # Explore data
    if args.explore or args.all_steps or args.all_fine_steps:
        logging.info("Exploring data")
        histogram_all(
            glb.data,
            glb.feature_names,
            glb.label_name,
            args.orientation,
            f"{glb.workingdir}/histograms.png"
        )
        scatter_all(
            glb.data,
            glb.feature_names,
            glb.label_name,
            args.orientation,
            f"{glb.workingdir}/scatters.png"
        )
    
    # Prepare data
    data_train = pd.DataFrame()
    data_test = pd.DataFrame()
    if args.prepare or args.all_steps:
        logging.info("Preparing data")
        data_train, data_test = split_data(glb)
    
    # Fit to Model
    regressor = None
    if args.fit or args.all_steps:
        try:
            if data_train.empty:
                data_train = pd.read_csv(glb.train_filename, dtype=glb.columns)
            if data_test.empty:
                data_test = pd.read_csv(glb.test_filename, dtype=glb.columns)
        except FileNotFoundError:
            logging.error(f"File not found: {glb.train_filename} and/or {glb.test_filename}")
            logging.error("Please run with -p flag to prepare data")
            sys.exit(1)
        
        x_train = data_train[glb.feature_names]
        y_train = data_train[glb.label_name]
        regressor = args.model
        
        logging.info("Fitting data to model...")
        regressor.fit(x_train, y_train)
        joblib.dump(regressor, glb.model_filename)
        logging.info("Model fit complete")
    
    # Use Scaler to Fit Data
    if args.fit_fine or args.all_fine_steps:
        try:
            if data_train.empty:
                data_train = pd.read_csv(glb.train_filename, dtype=glb.columns)
            if data_test.empty:
                data_test = pd.read_csv(glb.test_filename, dtype=glb.columns)
        except FileNotFoundError:
            logging.error(f"File not found: {glb.train_filename} and/or {glb.test_filename}")
            logging.error("Please run with -p flag to prepare data")
            sys.exit(1)
        
        x_train = data_train[glb.feature_names]
        print(x_train)
        y_train = data_train[glb.label_name]
        
        logging.info(f"Fitting data to model...")

        # scale data with x' = (x - u) / s
        scaler = sklearn.preprocessing.StandardScaler()
        logging.info("Scaling data strategy: StandardScaler")
        # find u and s
        scaler.fit(x_train)
        # transform data
        x_train_numpy = scaler.transform(x_train)
        # convert numpy array to DataFrame
        x_train = pd.DataFrame(x_train_numpy, columns=glb.feature_names)

        # peek at scaled data
        logging.debug("Scaled Features")
        logging.debug(glb.feature_names)
        logging.debug(x_train_numpy[:5,:])

        # do the fit/training
        regressor = args.model
        regressor.fit(x_train, y_train)

        # save the trained model
        joblib.dump((regressor,scaler), glb.model_filename)

        logging.info(f"Model fit complete")

    # Show Model & Function
    if args.show_model or args.all_steps:
        if regressor is None:
            try:
                regressor = joblib.load(glb.model_filename)
                if isinstance(regressor, tuple):
                    regressor = regressor[0]
            except FileNotFoundError:
                logging.error(f"File not found: {glb.model_filename}")
                logging.error("Please run with -f flag to fit model")
                sys.exit(1)

        try:
            # Show Model
            show_model(regressor)
        except Exception as e:
            logging.error(e)

        try:
            # Show Function
            show_function(regressor, glb)
        except Exception as e:
            logging.error(e)
    
    # Show Fine Model & Function
    if args.show_fine_model or args.all_fine_steps:
        if regressor is None:
            try:
                regressor, scaler = joblib.load(glb.model_filename)
            except FileNotFoundError:
                logging.error(f"File not found: {glb.model_filename}")
                logging.error("Please run with -f flag to fit model")
                sys.exit(1)

        try:
            # Show Fine Model
            show_fine_model(scaler, regressor)
        except Exception as e:
            logging.error(e)

        try:
            # Show Fine Function
            show_fine_function(glb, scaler, regressor)
        except Exception as e:
            logging.error(e)
    
    # Show Train Score
    if args.show_train_score or args.all_steps or args.all_fine_steps:
        try:
            if data_train.empty:
                data_train = pd.read_csv(glb.train_filename, dtype=glb.columns)
        except FileNotFoundError:
            logging.error(f"File not found: {glb.train_filename}")
            logging.error("Please run with -p flag to prepare data")
            sys.exit(1)

        if regressor is None:
            try:
                regressor = joblib.load(glb.model_filename)
            except FileNotFoundError:
                logging.error(f"File not found: {glb.model_filename}")
                logging.error("Please run with -f flag to fit model")
                sys.exit(1)

        if isinstance(regressor, tuple):
            regressor, scaler = regressor
            x_train = data_train[glb.feature_names]
            x_train = scaler.transform(x_train)
        else:
            x_train = data_train[glb.feature_names]
            y_train = data_train[glb.label_name]

        show_train_score(regressor, x_train, y_train)
        
        # Show Train Fit
        y_predicted = regressor.predict(x_train)
        filename = "scatters_fit_train.png"
        scatter_all_with_fit(
            data_train,
            glb.feature_names,
            glb.label_name,
            y_predicted,
            args.orientation,
            f"{glb.workingdir}{filename}"
        )
        print(f"Fit scatter plots saved in {glb.workingdir}{filename}")
        print()

    # Show Test Score
    if args.show_test_score or args.all_steps or args.all_fine_steps:
        try:
            if data_test.empty:
                data_test = pd.read_csv(glb.test_filename, dtype=glb.columns)
        except FileNotFoundError:
            logging.error(f"File not found: {glb.test_filename}")
            logging.error("Please run with -p flag to prepare data")
            sys.exit(1)

        if regressor is None:
            try:
                regressor = joblib.load(glb.model_filename)
            except FileNotFoundError:
                logging.error(f"File not found: {glb.model_filename}")
                logging.error("Please run with -f flag to fit model")
                sys.exit(1)

        # Show Test Score
        if isinstance(regressor, tuple):
            regressor, scaler = regressor
            x_test = data_train[glb.feature_names]
            x_test = scaler.transform(x_test)
        else:
            x_test = data_train[glb.feature_names]
            y_test = data_train[glb.label_name]

        show_test_score(glb)

        # Show Test Fit
        x_test = data_test[glb.feature_names]
        y_predicted = regressor.predict(x_test)
        filename = "scatters_fit_test.png"
        scatter_all_with_fit(
            data_test,
            glb.feature_names,
            glb.label_name,
            y_predicted,
            args.orientation,
            f"{glb.workingdir}{filename}"
        )
        print(f"Fit scatter plots saved in {glb.workingdir}{filename}")
        print()
    
    # Predict
    if args.predict:
        try:
            data_predict = pd.read_csv(args.predict, dtype=glb.columns)
        except FileNotFoundError:
            logging.error(f"File not found: {args.predict}")
            logging.error("Please provide a valid file")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error reading file: {args.predict}")
            logging.error(e)
            sys.exit(1)

        x_predict = data_predict[glb.feature_names]
        y_predicted = regressor.predict(x_predict)
        logging.debug("Predicted values:")
        logging.debug(y_predicted)

        filename = "scatters_fit_predict.png"
        scatter_all_with_fit(
            data_predict,
            glb.feature_names,
            glb.label_name,
            y_predicted,
            args.orientation,
            f"{glb.workingdir}{filename}"
        )
        print(f"Predicted scatter plots saved in {glb.workingdir}{filename}")
        print()

    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv)