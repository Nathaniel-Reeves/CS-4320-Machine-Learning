#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import math
import matplotlib.pyplot as plt

from globals import get_globals

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.debug("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features
    exclude_columns = my_args.exclude_columns

    if label_column in data.columns:
        label = label_column
    else:
        label = ""
    
    print("Exclude columns: {}".format(exclude_columns))

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in exclude_columns:
                continue
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column in exclude_columns:
                continue
            if feature_column != label:
                features.append(feature_column)

    return features, label

def display_feature_histograms(my_args, data, figure_number):
    """
    Display a histogram for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    size = int(math.ceil(math.sqrt(total_count)))
    
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Feature Histograms" )
    n_max = 1
    all_ax = []
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        if feature_column in data.columns:
            ax = fig.add_subplot(size, size, i)
            ax.set_yscale("log")
            n, bins, patches = ax.hist(data[feature_column], bins=20)
            if max(n) > n_max:
                n_max = max(n)
            ax.set_xlabel(feature_column)
            ax.locator_params(axis='x', tight=True, nbins=5)
            all_ax.append(ax)
        else:
            logging.warning("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))


    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.set_yscale("log")
        n, bins, patches = ax.hist(data[label_column], bins=20)
        if max(n) > n_max:
            n_max = max(n)
        ax.set_xlabel(label_column)
        ax.locator_params(axis='x', tight=True, nbins=5)
        all_ax.append(ax)

    for ax in all_ax:
        ax.set_ylim(bottom=1.0, top=n_max)

    fig.tight_layout()
    basename = get_basename(my_args.data_file)
    figure_name = "{}-{}-histogram.{}".format(basename, "-".join(feature_columns), "pdf")
    # fig.savefig(my_args.output_dir + '/' + figure_name)
    fig.savefig(my_args.output_dir + 'plots/histogram.pdf')
    plt.close(fig)
    return

def display_feature_histogram(glb, feature_column, figure_number):
    """
    Display a histogram for a single feature.
    """
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Feature Histogram" )
    ax = fig.add_subplot(1, 1, 1)
    ax.set_yscale("log")
    n, bins, patches = ax.hist(glb.data[feature_column], bins=20)
    ax.set_xlabel(feature_column)
    ax.locator_params(axis='x', tight=True, nbins=5)
    fig.tight_layout()
    basename = get_basename(glb.filename)
    figure_name = "{}-{}-histogram.{}".format(basename, feature_column, "pdf")
    fig.savefig(glb.out_dir + 'plots/' + figure_name)
    plt.close(fig)
    return

def display_label_vs_features(my_args, data, figure_number):
    """
    Display a plot of label vs feature for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    size = int(math.ceil(math.sqrt(total_count)))

    all_ax = []
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Label vs. Features" )
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        print("feature_column: {}".format(feature_column))
        if feature_column in data.columns:
            ax = fig.add_subplot(size, size, i)
            ax.scatter(feature_column, label_column, data=data, s=1)
            ax.set_xlabel(feature_column)
            ax.set_ylabel(label_column)
            ax.locator_params(axis='both', tight=True, nbins=5)
            all_ax.append(ax)
        else:
            logging.warning("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))
            
    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.scatter(label_column, label_column, data=data, s=1)
        ax.set_xlabel(label_column)
        ax.set_ylabel(label_column)
        ax.locator_params(axis='both', tight=True, nbins=5)
        all_ax.append(ax)

    fig.tight_layout()
    basename = get_basename(my_args.data_file)
    figure_name = "{}-{}-scatter.{}".format(basename, "-".join(feature_columns), "pdf")
    # fig.savefig(my_args.output_dir + '/' + figure_name)
    fig.savefig(my_args.output_dir + 'plots/scatter.pdf')
    plt.close(fig)

    return

def display_label_vs_feature(glb, feature_column, figure_number):
    """
    Display a plot of label vs feature for a single feature.
    """
    fig = plt.figure(figure_number, figsize=(6.5, 9))
    fig.suptitle( "Label vs. Feature" )
    ax = fig.add_subplot(1, 1, 1)
    rdf = reduce_dataframe(glb, feature_column)
    ax.scatter(feature_column, glb.label_name, data=rdf, s=1)
    ax.set_xlabel(feature_column)
    ax.set_ylabel(glb.label_name)
    ax.locator_params(axis='both', tight=True, nbins=5)
    fig.tight_layout()
    basename = get_basename(glb.filename)
    figure_name = "{}-{}-scatter.{}".format(basename, feature_column, "pdf")
    fig.savefig(glb.out_dir + 'plots/' + figure_name)
    plt.close(fig)
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Create Data Plots')
    parser.add_argument(
        'action',
        default='all',
        choices=[ "label-vs-features", "feature-histograms", "all" ],
        nargs='?',
        help="desired action"
    )
    parser.add_argument(
        '--features',
        '-f',
        default=None,
        action="extend",
        nargs="+",
        type=str,
        help="column names for features"
    )
    parser.add_argument(
        '--label',
        '-l',
        default="label",
        type=str,
        help="column name for label"
    )
    parser.add_argument(
        '--exclude-columns',
        '-e',
        default=[],
        action="extend",
        nargs="+",
        type=str,
        help="columns to exclude"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="out",
        type=str,
        help="output directory for plots"
    )

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def reduce_dataframe(glb, feature_name):
    # reduced dataframe
    df2 = glb.data[[feature_name, glb.label_name]].dropna()
    
    # print diff in number rows between original and reduced dataframes
    logging.info("Original data frame: {}".format(glb.data.shape))
    logging.info("Reduced data frame: {}".format(df2.shape))
    
    return df2

# python3 display_data.py -l "SalePrice" -d "data/train.csv" -e "Id" -f "LotArea"

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    
    glb = get_globals()
    
    for feature_name in glb.feature_names:
        
        logging.info("feature_name: {}".format(feature_name))
        
        if my_args.action in ("all", "label-vs-features"):
            try:
                display_label_vs_feature(glb, feature_name, 1)
            except Exception as e:
                logging.error("Exception: {}".format(e))
        if my_args.action in ("all", "feature-histograms"):
            try:
                display_feature_histogram(glb, feature_name, 2)
            except Exception as e:
                logging.error("Exception: {}".format(e))
    return

if __name__ == "__main__":
    main(sys.argv)
    
