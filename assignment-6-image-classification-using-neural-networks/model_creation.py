#!/usr/bin/env python3

#
# Keep the model creation code contained here
#
import tensorflow as tf
import keras

def create_model(my_args, input_shape):
    """
    Control function.
    Selects the correct function to build a model, based on the model name
    from the command line arguments.

    Assumes my_args.model_name is set.
    """
    create_functions = {
        "a": create_model_a,
        "b": create_model_b,
        "c": create_model_c,
        "d": create_model_d,
        "e": create_model_e,
        "f": create_model_f
    }
    if my_args.model_name not in create_functions:
        raise ValueError("Invalid model name: {} not in {}".format(my_args.model_name, list(create_functions.keys())))
        
    model = create_functions[my_args.model_name](my_args, input_shape)
    print(model.summary())
    return model


### Various model architectures

def create_model_a(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam())
    # Epoch 1 val_accuracy: 0.2865
    # Score to Beet val_accuracy: 0.4100 epoch:9/10 Batch: 1
    return model

def create_model_b(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam())
    # Epoch 1 val_accuracy: 0.0945
    # Score to Beet val_accuracy: 0.3950 epoch:8/10 Batch: 1
    return model

def create_model_c(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(13,13), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(9,9), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(32, activation="relu", kernel_initializer="he_normal"))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam())
    # Epoch 1 val_accuracy: 0.1045
    # Score to Beet val_accuracy: 0.3950 epoch:8/10 Batch: 1
    return model

def create_model_d(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam())
    # Score to Beet val_accuracy: 0.5355 epoch:12/100 Batch: All
    return model

def create_model_e(my_args, input_shape):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)
    # Score to Beet val_accuracy: 0.4998 epoch:8/100 Batch: 1
    return model

def create_model_f(my_args, input_shape):
    # https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu", kernel_initializer="he_uniform", padding="same"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu", kernel_initializer="he_uniform"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation="softmax"))
    opt = keras.optimizers.SGD(learning_rate=0.001, ema_momentum=0.9)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=opt)
    # Score to Beet val_accuracy: 0.4998 epoch:8/100 Batch: 1
    return model