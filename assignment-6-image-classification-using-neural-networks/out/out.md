# Model 1
tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_normal"))
model.add(keras.layers.Dense(1000, activation="elu", kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"))
model.add(keras.layers.Dense(1000, activation="elu", kernel_initializer="he_normal"))
model.add(keras.layers.BatchNormalization()) 
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_normal"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))

loss = "mean_squared_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(),
              metrics=["R2Score"])

Epoch 9/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 54s 2ms/step - R2Score: -0.0025 - loss: 753107.1250 - val_R2Score: -5.9247e-04 - val_loss: 742566.0625 - learning_rate: 0.0100

Kaggle Score
Private: 1.16797
Public:  1.16375

# Model 2
tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_normal"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))

loss = "mean_squared_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(),
              metrics=["R2Score"])

Epoch 11/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 30s 948us/step - R2Score: -3.4924e-04 - loss: 748922.6250 - val_R2Score: -5.7578e-05 - val_loss: 742170.1875 - learning_rate: 0.0100

Kaggle Score
Private: 1.17257
Public:  1.16833

# Model 3
tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))

loss = "mean_squared_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(clipvalue=1.0),
              metrics=["R2Score"])

Epoch 8/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 43s 1ms/step - R2Score: -1.2104e-04 - loss: 750087.1875 - val_R2Score: -9.9540e-05 - val_loss: 742199.4375 - learning_rate: 0.0100

Kagle Score
Private: 1.17256
Public:  1.16831

# Model 4
tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))
#print(model.summary())
#print(model.layers[1].get_weights())


loss = "mean_squared_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(clipvalue=1.0),
              metrics=["R2Score"])

Epoch 13/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 34s 1ms/step - R2Score: -2.2000e-04 - loss: 752053.0625 - val_R2Score: -0.0014 - val_loss: 743169.6250 - learning_rate: 0.0099

Kagle Score
Private: 1.17239
Public:  1.16815

# Model 5
tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(100, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(10, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))
#print(model.summary())
#print(model.layers[1].get_weights())

#
# Compile the model
#
loss = "mean_squared_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(clipvalue=1.0),
              metrics=["R2Score"])

Epoch 11/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 29s 905us/step - R2Score: -0.0014 - loss: 751084.0625 - val_R2Score: -0.0013 - val_loss: 743105.5625 - learning_rate: 0.0100

Kagle Score
Private: 1.17641
Public:  1.17214

# Model 6

tf.random.set_seed(42)
model = keras.Sequential()
model.add(keras.layers.Input(shape=input_shape))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1000, activation="sigmoid", kernel_initializer="glorot_uniform"))
model.add(keras.layers.Dense(1, activation="linear", kernel_initializer="glorot_uniform"))

loss = "mean_squared_logarithmic_error"
model.compile(loss=loss,
              optimizer=keras.optimizers.SGD(clipvalue=1.0),
              metrics=["R2Score"])

Epoch 24/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 169s 6ms/step - R2Score: -0.1864 - loss: 1.2005 - val_R2Score: -0.1878 - val_loss: 1.1906 - learning_rate: 0.0093
Epoch 25/100
30000/30000 ━━━━━━━━━━━━━━━━━━━━ 168s 6ms/step - R2Score: -0.1856 - loss: 1.2053 - val_R2Score: -0.1877 - val_loss: 1.1906 - learning_rate: 0.0093

Kaggle Score
Private: 1.09711
Public:  1.09421