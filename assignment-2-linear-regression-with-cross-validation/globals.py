#!/usr/bin/env python3

import pandas as pd
import logging
from dataclasses import dataclass, field

###########################################################
# Set global values for filenames, features, label
# Load data
###########################################################
_columns = {
    "Socioeconomic Score": float,
    "Study Hours": float,
    "Sleep Hours": float,
    "Attendance (%)": float,
    "Grades": float
}

_label_name = "Grades"
_filename = "data.csv"
_train_filename = "data-train.csv"
_test_filename = "data-test.csv"
_model_filename = "GradesModel.joblib"

_ratio = 0.2
_seed = 42
_dir = "data/"

@dataclass
class Globals:
    data: pd.DataFrame = pd.DataFrame()
    label_name: str = field(default_factory=str)
    filename: str = "data/data.csv"
    train_filename: str = "data/data-train.csv"
    test_filename: str = "data/data-test.csv"
    model_filename: str = "data/Data.joblib"
    columns: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    ratio: float = 0.2
    seed: int = 42
    workingdir: str = "data/"

    def initialize(self):
        self.fetch_data()
        self.set_feature_names()
    
    def set_feature_names(self):
        if not self.feature_names:
            self.feature_names = list(self.columns.keys())
            self.feature_names.remove(self.label_name)
        return self.feature_names
    
    def fetch_data(self):
        if self.data.empty:
            try:
                self.data = pd.read_csv(self.filename, dtype=self.columns)
            except FileNotFoundError:
                logging.error(f"Data file not found: {self.filename}")
                exit(1)
        return self.data
            
def get_globals():
    glb = Globals(
        columns=_columns,
        label_name=_label_name,
        filename= _dir + _filename,
        train_filename= _dir +_train_filename,
        test_filename= _dir +_test_filename,
        model_filename= _dir +_model_filename,
        ratio=_ratio,
        seed=_seed,
        workingdir=_dir
    )
    return glb
