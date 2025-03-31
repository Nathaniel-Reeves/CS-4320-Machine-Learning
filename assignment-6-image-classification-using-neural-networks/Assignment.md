Image Classification using Convolutional Neural Network
-------------------------------------------------------

In this assignment, you will fit a convolutional neural network
for image classification tasks.

### Caution!!!!

Because the data is a well known and popular dataset
for learning and improving skills with machine learning,
the testing data is part of the downloads.
*Best practices are to keep the testing data until after training has completed.*
You should only judge your best model on the test data when
you are ready to write your final report.

# Fit a Convolutional Neural Network

## Dataset 

[CIFAR10](https://keras.io/api/datasets/cifar10/)

The data set is called "CIFAR-10" and is available through Keras.
The images are 32x32 RGB color images in 10 categories.

There are 50,000 training images. Each image has 32x32x3x1 = 3072 bytes.
This is about 150 MB for the entire training data set. The dataset should
not be too large for your computer. However, if it is, you can train
on a smaller subset of the data, save the model, then continue training
on a different subset. 


## Model

Vision network, trained from scratch, using Keras and TensorFlow.

You should start with a basic vision network structure, and modify
the details as part of your training process. Suggested layer types
are:

- [Conv2D](https://keras.io/api/layers/convolution_layers/convolution2d/)
- [MaxPooling2D](https://keras.io/api/layers/pooling_layers/max_pooling2d/)
- [Dense](https://keras.io/api/layers/core_layers/dense/)
- [Dropout](https://keras.io/api/layers/regularization_layers/dropout/)
- [Flatten](https://keras.io/api/layers/reshaping_layers/flatten/)

The network architecture, the activation functions, the kernel
initialization, and the optimizer are all hyperparameters that can
be tuned.

Pay attention to memory usage on your system. Too many parameters will
cause the model not to fit in memory.

## Training

Select an optimizer and a learning rate. Consider the following
hyperparameters:

- Optimizer
- Batch size
- Number of epochs
- Learning rate
- Learning rate decay

## Loss

Use the loss function that works best for you. Remember that this
guides the learning process. It is a hyperparameter that can
be tuned.

## Metric

Your final model will be measured by the accuracy of the model
on the validation and test data. 

### Accuracy vs CategoricalAccuracy

Both are in range [0, 1], computed as # correct / # total.

To use Accuracy, y_pred and y_true must both be the class number. So, each instance is a class number.

To use CategoricalAccuracy, y_pred and y_true must both be one-hot-encoded. So, each instance is a vector of probabilities, and the predicted class is the index with the highest probability.

[Accuracy, "accuracy"](https://keras.io/api/metrics/accuracy_metrics/#accuracy-class)

[CategoricalAccuracy, "categorical_accuracy"](https://keras.io/api/metrics/accuracy_metrics/#categoricalaccuracy-class)

### Benchmark

Best practices are to get the best validation score you can get,
expecting it to reflect the best model performance on the test set.
Our benchmark then is to obtain a good validation score, and
that the test score is close to the validation score.

Note that the training score is not a good benchmark. If you
have a sufficiently large model, it can eventually memorize
all the training data and achieve a near perfect score.

Here, we quantify what we mean by 'good' and 'close'.

- Training data accuracy: *whatever*
- Validation data accuracy ('val_categorical_accuracy'): > 0.75
- Testing data accuracy ('test_categorical_accuracy'): > 0.75
- abs(val_categorical_accuracy - test_categorical_accuracy): < 0.03


# Documentaion

Your report should contain a summary of the model architecture,
the hyperparameters used, and the results of your training.

## Architecture

All layers and their configurations should be specified,
such that anyone reading the report can create an identical model.

## Hyperparameters

All hyperparameter choices not documented in the architecture
should be documented here. Anyone reading the report should
read this section and be able to train their model in the
identical way that you trained yours.

## Results

Help the reader understand the quality of predictions to
expect when training a model like yours in the way you
trained yours.

### Training Results

Learning curve plot(s) comparing the loss and metrics obtained
on a per-epoch basis during training, with both training and
validation data included.

Include the confusion matrix on the training and/or validation
data of your final model. Either a tabular form or a graphical form
would be appropriate.


### Final Results

A table of results should show the loss and metric values
obtained for the best model. This includes the training, validation
and testing results. Note there is no loss value for the
testing data.

## Conclusions

Make a recommendation regarding your model. Should it be used
for production? Why or why not?

# Code

There is a simple starter code base that works (poorly) on the
"Fashion MNIST" dataset. You may start with it and make modifications
to improve your process.