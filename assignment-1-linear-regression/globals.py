#!/usr/bin/env python3

import pandas as pd

###########################################################
# Set global values for filenames, features, label
# Load data
###########################################################
columns = {
    "Socioeconomic Score": float,
    "Study Hours": float,
    "Sleep Hours": float,
    "Attendance (%)": float,
    "Grades": float
}

label_name = "Grades"
filename = "data/data.csv"
train_filename = "data/data-train.csv"
test_filename = "data/data-test.csv"
model_filename = "data/Grades.joblib"

ratio = 0.2
seed = 42

feature_names = list(columns.keys())
feature_names.remove(label_name)

data = pd.read_csv(filename, dtype=columns)
X_train = data[feature_names]
y_train = data[label_name]
