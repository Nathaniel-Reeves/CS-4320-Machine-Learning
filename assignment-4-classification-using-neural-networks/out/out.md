# Default settings from class starter code.

tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

loss = "binary_crossentropy"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(learning_rate=0.1),
              metrics=["AUC"])

early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

Epoch 8/100
3518/3518 ━━━━━━━━━━━━━━━━━━━━ 4s 972us/step - AUC: 0.9746 - loss: 0.1506 - val_AUC: 0.9749 - val_loss: 0.1515 - learning_rate: 0.1000

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Layer (type)     ┃ Output Shape    ┃   Param # ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ dense (Dense)    │ (None, 100)     │    28,000 │
├──────────────────┼─────────────────┼───────────┤
│ dense_1 (Dense)  │ (None, 1)       │       101 │
└──────────────────┴─────────────────┴───────────┘
 Total params: 28,103 (109.78 KB)
 Trainable params: 28,101 (109.77 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)

Kaggle Prediction Scores: Private 0.93871  Public 0.94051

# Try few wide layers

tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
#print(model.summary())
#print(model.layers[1].get_weights())

loss = "binary_crossentropy"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(learning_rate=0.1),
              metrics=["AUC"])

early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

Epoch 8/100
3518/3518 ━━━━━━━━━━━━━━━━━━━━ 8s 2ms/step - AUC: 0.9761 - loss: 0.1461 - val_AUC: 0.9750 - val_loss: 0.1531 - learning_rate: 0.1000

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Layer (type)     ┃ Output Shape    ┃   Param # ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ dense (Dense)    │ (None, 500)     │   140,000 │
├──────────────────┼─────────────────┼───────────┤
│ dense_1 (Dense)  │ (None, 500)     │   250,500 │
├──────────────────┼─────────────────┼───────────┤
│ dense_2 (Dense)  │ (None, 500)     │   250,500 │
├──────────────────┼─────────────────┼───────────┤
│ dense_3 (Dense)  │ (None, 1)       │       501 │
└──────────────────┴─────────────────┴───────────┘
 Total params: 641,503 (2.45 MB)
 Trainable params: 641,501 (2.45 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)

Kaggle Prediction Scores: Private 0.93933  Public 0.94109

# Try many thin layers

tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
#print(model.summary())
#print(model.layers[1].get_weights())

loss = "binary_crossentropy"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(learning_rate=0.1),
              metrics=["AUC"])

early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

Epoch 8/100
3518/3518 ━━━━━━━━━━━━━━━━━━━━ 5s 1ms/step - AUC: 0.9749 - loss: 0.1495 - val_AUC: 0.9735 - val_loss: 0.1562 - learning_rate: 0.1000

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)      ┃ Output Shape  ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ dense (Dense)     │ (None, 100)   │     28,000 │
├───────────────────┼───────────────┼────────────┤
│ dense_1 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_2 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_3 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_4 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_5 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_6 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_7 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_8 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_9 (Dense)   │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_10 (Dense)  │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_11 (Dense)  │ (None, 100)   │     10,100 │
├───────────────────┼───────────────┼────────────┤
│ dense_12 (Dense)  │ (None, 1)     │        101 │
└───────────────────┴───────────────┴────────────┘
 Total params: 139,203 (543.77 KB)
 Trainable params: 139,201 (543.75 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)

Kaggle Prediction Scores: Private 0.93936  Public 0.94077


# Improve on the best strategy,  Try random null layers and other stuff discussed in class today

Decided to use the Few Wide layers since there wasn't much difference between Few Wide Layers and Many Thin Layers and because it has less hyperperamters.

tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(500, activation="leaky_relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(500, activation="leaky_relu", kernel_initializer="he_normal"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

loss = "binary_crossentropy"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(learning_rate=0.1, clipvalue=1.0, momentum=0.8, nesterov=True),
              metrics=["AUC"])

early_stop_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

Epoch 30/30
3518/3518 ━━━━━━━━━━━━━━━━━━━━ 8s 2ms/step - AUC: 0.9831 - loss: 0.1257 - val_AUC: 0.9665 - val_loss: 0.1839 - learning_rate: 0.0905

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Layer (type)      ┃ Output Shape   ┃   Param # ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ dense (Dense)     │ (None, 500)    │   140,000 │
├───────────────────┼────────────────┼───────────┤
│ dense_1 (Dense)   │ (None, 500)    │   250,500 │
├───────────────────┼────────────────┼───────────┤
│ dense_2 (Dense)   │ (None, 1)      │       501 │
└───────────────────┴────────────────┴───────────┘
 Total params: 782,004 (2.98 MB)
 Trainable params: 391,001 (1.49 MB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 391,003 (1.49 MB)

Kaggle Prediction Scores: Private 0.93936  Public 0.94077

## Activation Function Notes

1. Linear (No-op) (keras default)
    - Random Type: Glorot
2. Sigmoid (sigmoid)
    - Random Type: Glorot
3. Rectified Linear (relu)
    - Not Smooth, Zero Slope (Neurons "die")
    - Random Type: He
4. Leaky ReLU (leaky_relu)
    - Small Gradient (Neurons dont "die")
    - Random Type: He
5. Exponential Liner Unit (elu)
    - Smooth if alpha = 1,
    - Slower to compute
    - Small gradient on left side (Neurons dont "die")
    - Better gradients at the cost of more compute
    - Random Type: He
6. Scaled Elu (selu)
    - Only useful on stacks of dense layers.
    - Same size layers and densly connected (all neurons connect to all neurons between layers.)
    - Hyper perameter of (s) or scale value.
    - Not great if any other layer configuration is set.
    - Random Type: LeCun
7. Gaussian ELU (gelu)
    - Maybe faster convergence
    - Uses Gaussian Cumulative Distribution Function
    - More compute (expensive)
    - Random Type: He
8. Sigmoid Linear Unit or Swish  (silu, swish)
    - the Sigmoid Function
    - sometimes better than GELU
    - Random Type: He
9. MISH (mish)
    - the hyperbolic tangent function of the log of 1 + e to the x
    - Even more compute (expensive)
    - Random Type: He

```
model.add(keras.layers.Dense(units, activation="leaky_relu", kernel_initializer="he_normal"))
```

## Initializations (Types of Random)
Function                | Keras Name
Zeros                   | zeros
Ones                    | ones
Random Normal           | random_normal
Random Uniform          | random_uniform
Glorot (Xavier) Normal  | glorot_normal
Glorot (Xavier) Uniform | glorot_uniform
He Normal               | he_normal
He Uniform              | he_uniform
Lecun Normal            | lecun_normal
Lecun Uniform           | lecun_uniform

normal = normal distribution (bell curve)
uniform = equal probability

## Batch Normalization
Try rescaling data before each layer.
```
# Batch normalization layers before each neuron layer
model.add(keras.layers.BatchNormalization()) 

# Add Layer
model.add(keras.layers.Dense(units, ...))

# Add Batch Normalization
...

# Add Layer
...
```

## Gradient Cliping
clip extreme outputs from neurons back to a normal value
```
# Add as an optimizer to keras

keras.optimizers.SGD(clipvalue=1.0)
# OR
keras.optimizers.SGD(clipnorm=1.0)
```

clip value sets the extreme value to 1
clip norm estimates what the value should be (more compute)

## Try using this too
keras.optimizers.SGD(momentum=0.9, nestrov=True)
