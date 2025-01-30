import sys
import argparse
import logging

from globals import (
    data,
    feature_names,
    label_name,
    train_filename,
    test_filename,
    ratio,
    seed
)
from helpers.explore_data import histogram_all, scatter_all
from helpers.split_data import split_data

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Linear Regression')
    parser.add_argument(
        "--logging-level",
        "-l",
        type=str,
        help="logging level: warn, info, debug",
        choices=("warn", "info", "debug"),
        default="warn",
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
        default=False
    )
    parser.add_argument(
        "--prepare",
        "-p",
        help="prepare: clean, transform",
        default=False
    )

    my_args = parser.parse_args(argv[1:])
    if my_args.logging_level == "warn":
        my_args.logging_level = logging.WARN
    elif my_args.logging_level == "info":
        my_args.logging_level = logging.INFO
    elif my_args.logging_level == "debug":
        my_args.logging_level = logging.DEBUG

    return my_args

def main(argv):
    # parse command line arguments, Set logging level
    args = parse_args(argv)
    logging.basicConfig(level=args.logging_level)

    # first few rows of the DataFrame
    logging.debug(f"data.csv HEAD\n {data.head()}")
    
    # Explore data
    explore = False
    if args.explore:
        logging.info("Exploring data")
        histogram_all(data, feature_names, label_name, args.orientation, "data/histograms.png")
        scatter_all(data, feature_names, label_name, args.orientation , "data/scatters.png")
    
    # Prepare data
    prepare = True
    if prepare:
        logging.info("Preparing data")
        split_data(data, train_filename, test_filename, seed, ratio)
    

if __name__ == "__main__":
    main(sys.argv)