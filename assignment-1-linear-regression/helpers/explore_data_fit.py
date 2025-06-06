#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import math
import joblib

def pdf_figure(figure_number):
    """Configure figure for portrait orientation paper"""
    # letter paper dimensions
    width = 6.5
    height = 9
    fig = plt.figure(figure_number, figsize=(width, height))
    return fig

def png_figure(figure_number):
    """Configure figure for landscape orientation screen"""
    # 16:9 ratio, on paper dimensions
    width = 9
    height = 5
    fig = plt.figure(figure_number, figsize=(width, height))
    return fig

def scatter_column(fig, feature_series, label_series, plot_count, plot_number):
    """
    Use the feature values as the x-axis, and the label as the y-axis.
    Scatter plot the data in the new axes created here.
    """
    ax = fig.add_subplot(plot_count, plot_count, plot_number)
    ax.scatter(feature_series, label_series, s=1, label="data")
    ax.set_xlabel(feature_series.name)
    ax.set_ylabel(label_series.name)
    ax.locator_params(axis='both', tight=True, nbins=5)
    return ax

def scatter_all_with_fit(data, feature_names, label_name, y_predicted, orientation="landscape", figure_name="scatters_with_fit.png"):
    """
    For each feature, scatter plot it vs the label.
    Also plot it vs the predicted label value
    """
    figure_number = 2
    if orientation == "portrait":
        fig = pdf_figure(figure_number)
    else:
        fig = png_figure(figure_number)
    fig.suptitle("Features vs. Label")

    column_count = len(feature_names) + 1
    plot_count = int(math.ceil(math.sqrt(column_count)))
    plot_number = 1
    all_ax = []
    for column_name in feature_names + [label_name]:
        ax = scatter_column(fig, data[column_name], data[label_name], plot_count, plot_number)
        ax.scatter(data[column_name], y_predicted, s=1, color="magenta", label="fit")
        ax.legend()
        all_ax.append(ax)
        plot_number += 1

    fig.tight_layout()
    fig.savefig(figure_name)
    plt.close(fig)
    return
