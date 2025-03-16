#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.metrics
import joblib
import tensorflow as tf
import tensorflow.keras as keras

import image_functions
from cnn_common import *

def load_model(my_args):
    train_file = my_args.train_file
    model_file = get_model_filename(my_args.model_file, train_file)
    if not os.path.exists(model_file):
        raise Exception("Model file, '{}', does not exist.".format(model_file))
    pipeline = joblib.load(model_file)
    (pipeline, model) = pipeline

    return pipeline, model

def load_and_transform_data(my_args):
    train_file = my_args.train_file
    if not os.path.exists(train_file):
        raise Exception("training data file: {} does not exist.".format(train_file))
    
    pipeline, model = load_model(my_args)
    X, y = load_data(my_args, train_file)
    
    X = pipeline.transform(X) # If the resulting array is sparse, use .todense()
    # reshape the 784 pixels into a 2D greyscale image
    X = np.reshape(X,[X.shape[0],28,28,1])
    return X, y

def do_text_model_view(my_args):
    pipeline, model = load_model(my_args)
    print(model.summary())
    return

def do_text_model_slice_view(my_args):
    pipeline, model = load_model(my_args)
    model = keras.models.Sequential(model.layers[0:my_args.layer])
    print(model.summary())
    return

def do_text_data_view(my_args):
    X, y = load_and_transform_data(my_args)
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    return

def do_text_instance_view(my_args):
    X, y = load_and_transform_data(my_args)
    i = my_args.instance
    print("X[{}]:".format(i), X[i,:,:,0])
    print("y[{}]:".format(i), y[i])
    print("X[{}].shape:".format(i), X[i,:,:].shape)
    print("y[{}].shape:".format(i), y[i].shape)
    return

def do_png_instance_view(my_args):
    X, y = load_and_transform_data(my_args)
    i = my_args.instance
    image_data = X[i]
    image_class = y[i]
    image_name = "instance-i{:04d}-c{:02d}.png".format(i, image_class)
    image_functions.save_image(image_name, image_data)
    print("Saved instance {} as {}.".format(i, image_name))
    return

def do_text_instance_filter_view(my_args):
    pipeline, model = load_model(my_args)
    model = keras.models.Sequential(model.layers[0:my_args.layer])

    X, y = load_and_transform_data(my_args)
    i = my_args.instance
    image_data = X[i]
    image_class = y[i]

    image_data = np.reshape(image_data, [-1, 28, 28, 1])
    filter_output = model.predict(image_data)
    print("filter[{}]:".format(my_args.filter), filter_output[0,:,:,my_args.filter])
    return

def do_png_instance_filter_view(my_args):
    pipeline, model = load_model(my_args)
    model = keras.models.Sequential(model.layers[0:my_args.layer])

    X, y = load_and_transform_data(my_args)
    i = my_args.instance
    image_data = X[i]
    image_class = y[i]

    image_data = np.reshape(image_data, [-1, 28, 28, 1])
    filter_output = model.predict(image_data)

    filter_data = filter_output[0,:,:,my_args.filter:my_args.filter+1]
    image_name = "filter-i{:04d}-c{:02d}-l{:02d}-f{:03d}.png".format(i, image_class, my_args.layer, my_args.filter)
    image_functions.save_image(image_name, filter_data)
    print("Saved filter {} as {}.".format(my_args.filter, image_name))

    return

def do_png_instance_layer_view(my_args):
    pipeline, model = load_model(my_args)
    model = keras.models.Sequential(model.layers[0:my_args.layer])

    X, y = load_and_transform_data(my_args)
    i = my_args.instance
    image_data = X[i]
    image_class = y[i]

    image_data = np.reshape(image_data, [-1, 28, 28, 1])
    filter_output = model.predict(image_data)

    count = filter_output.shape[3]
    size = 1
    while size*size < count:
        size+=1

    row_size = filter_output.shape[1]
    col_size = filter_output.shape[2]
    height = row_size * size
    width = col_size * size
    layer_data = np.zeros([height, width, 1])
    for row in range(size):
        for col in range(size):
            if row*size+col < count:
                layer_data[row*row_size:(row+1)*row_size, col*col_size:(col+1)*col_size,0] = filter_output[0,:,:,row*size+col]
    
    

    # filter_data = filter_output[0,:,:,my_args.filter:my_args.filter+1]
    image_name = "layer-i{:04d}-c{:02d}-l{:02d}.png".format(i, image_class, my_args.layer)
    image_functions.save_image(image_name, layer_data)
    print("Saved layer {} as {}.".format(my_args.layer, image_name))

    return


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image classification model visualizer')
    parser.add_argument('action', default='text',
                        choices=[ "text", "model-slice",
                                  "text-data",
                                  "text-instance", "png-instance",
                                  "text-filter", "png-filter", "png-layer" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--train-file',    '-t', default="",    type=str,   help="name of file with training data")
    parser.add_argument('--test-file',     '-T', default="",    type=str,   help="name of file with test data (default is constructed from train file name)")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--features',      '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',         '-l', default="label",   type=str,   help="column name for label")

    parser.add_argument('--instance',      '-i', default=0,         type=int,   help="data instance to use (default=0)")
    parser.add_argument('--layer',         '-L', default=1,         type=int,   help="number of model layers to use (default=1)")
    parser.add_argument('--filter',        '-F', default=0,         type=int,   help="which filter output to use (default=0)")


    parser.add_argument('--shuffle',                       action='store_true',  help="Shuffle data when loading.")
    parser.add_argument('--no-shuffle',    dest="shuffle", action='store_false', help="Do not shuffle data when loading.")
    parser.set_defaults(shuffle=True)

    parser.add_argument('--logging',     
                        default="warn",
                        type=str,  
                        choices=("warn", "info", "debug"),
                        help="Level of logging to apply. default=(warn).")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    if my_args.logging == "warn":
        my_args.logging = logging.WARN
    elif my_args.logging == "info":
        my_args.logging = logging.INFO
    elif my_args.logging == "debug":
        my_args.logging = logging.DEBUG
    else:
        raise Exception("Unexpected value of --logging {}".format(my_args.logging))
    
    return my_args


def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=my_args.logging)

    if my_args.action == 'text':
        do_text_model_view(my_args)
    elif my_args.action == 'model-slice':
        do_text_model_slice_view(my_args)
    elif my_args.action == 'text-data':
        do_text_data_view(my_args)
    elif my_args.action == 'text-instance':
        do_text_instance_view(my_args)
    elif my_args.action == 'png-instance':
        do_png_instance_view(my_args)
    elif my_args.action == 'text-filter':
        do_text_instance_filter_view(my_args)
    elif my_args.action == 'png-filter':
        do_png_instance_filter_view(my_args)
    elif my_args.action == 'png-layer':
        do_png_instance_layer_view(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    
