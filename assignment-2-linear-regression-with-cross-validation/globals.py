#!/usr/bin/env python3
import logging
import os
from dataclasses import dataclass, field
import pandas as pd
import sklearn.pipeline
import sklearn.base

###########################################################
# Set global values for filenames, features, label
# Load data
###########################################################
_columns = {
    "Id": int,
    "MSSubClass": (20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190),
    "MSZoning": ("A", "C", "FV", "I", "RH", "RL", "RP", "RM", "C (all)"),
    "LotFrontage": float,
    "LotArea": float,
    "Street": ("Grvl", "Pave"),
    "Alley": ("Grvl", "Pave", "NA"),
    "LotShape": ("Reg", "IR1", "IR2", "IR3"),
    "LandContour": ("Lvl", "Bnk", "HLS", "Low"),
    "Utilities": ("AllPub", "NoSewr", "NoSeWa", "ELO"),
    "LotConfig": ("Inside", "Corner", "CulDSac", "FR2", "FR3"),
    "LandSlope": ("Gtl", "Mod", "Sev"),
    "Neighborhood": ("Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "Names", "NoRidge", "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker", "NAmes"),
    "Condition1": ("Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"),
    "Condition2": ("Artery", "Feedr", "Norm", "RRNn", "RRAn", "PosN", "PosA", "RRNe", "RRAe"),
    "BldgType": ("1Fam", "2FmCon", "Duplx", "TwnhsE", "TwnhsI", "Twnhs", "Duplex", "2fmCon"),
    "HouseStyle": ("1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"),
    "OverallQual": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "OverallCond": (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    "YearBuilt": int,
    "YearRemodAdd": int,
    "RoofStyle": ("Flat", "Gable", "Gambrel", "Hip", "Mansard", "Shed"),
    "RoofMatl": ("ClyTile", "CompShg", "Membran", "Metal", "Roll", "Tar&Grv", "WdShake", "WdShngl"),
    "Exterior1st": ("AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing", "Wd Shng", "Brk Cmn", "CmentBd"),
    "Exterior2nd": ("AsbShng", "AsphShn", "BrkComm", "BrkFace", "CBlock", "CemntBd", "HdBoard", "ImStucc", "MetalSd", "Other", "Plywood", "PreCast", "Stone", "Stucco", "VinylSd", "Wd Sdng", "WdShing", "Wd Shng", "Brk Cmn", "CmentBd"),
    "MasVnrType": ("BrkCmn", "BrkFace", "CBlock", "None", "Stone"),
    "MasVnrArea": float,
    "ExterQual": ("Ex", "Gd", "TA", "Fa", "Po"),
    "ExterCond": ("Ex", "Gd", "TA", "Fa", "Po"),
    "Foundation": ("BrkTil", "CBlock", "PConc", "Slab", "Stone", "Wood"),
    "BsmtQual": ("Ex", "Gd", "TA", "Fa", "Po", "NA"),
    "BsmtCond": ("Ex", "Gd", "TA", "Fa", "Po", "NA"),
    "BsmtExposure": ("Gd", "Av", "Mn", "No", "NA"),
    "BsmtFinType1": ("GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"),
    "BsmtFinSF1": float,
    "BsmtFinType2": ("GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "NA"),
    "BsmtFinSF2": float,
    "BsmtUnfSF": float,
    "TotalBsmtSF": float,
    "Heating": ("Floor", "GasA", "GasW", "Grav", "OthW", "Wall"),
    "HeatingQC": ("Ex", "Gd", "TA", "Fa", "Po"),
    "CentralAir": ("N", "Y"),
    "Electrical": ("SBrkr", "FuseA", "FuseF", "FuseP", "Mix"),
    "1stFlrSF": float,
    "2ndFlrSF": float,
    "LowQualFinSF": float,
    "GrLivArea": float,
    "BsmtFullBath": float,
    "BsmtHalfBath": float,
    "FullBath": float,
    "HalfBath": float,
    "BedroomAbvGr": float,
    "KitchenAbvGr": float,
    "KitchenQual": ("Ex", "Gd", "TA", "Fa", "Po"),
    "TotRmsAbvGrd": float,
    "Functional": ("Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"),
    "Fireplaces": float,
    "FireplaceQu": ("Ex", "Gd", "TA", "Fa", "Po", "NA"),
    "GarageType": ("2Types", "Attchd", "Basment", "BuiltIn", "CarPort", "Detchd", "NA"),
    "GarageYrBlt": int,
    "GarageFinish": ("Fin", "RFn", "Unf", "NA"),
    "GarageCars": float,
    "GarageArea": float,
    "GarageQual": ("Ex", "Gd", "TA", "Fa", "Po", "NA"),
    "GarageCond": ("Ex", "Gd", "TA", "Fa", "Po", "NA"),
    "PavedDrive": ("Y", "P", "N"),
    "WoodDeckSF": float,
    "OpenPorchSF": float,
    "EnclosedPorch": float,
    "3SsnPorch": float,
    "ScreenPorch": float,
    "PoolArea": float,
    "PoolQC": ("Ex", "Gd", "TA", "Fa", "NA"),
    "Fence": ("GdPrv", "MnPrv", "GdWo", "MnWw", "NA"),
    "MiscFeature": ("Elev", "Gar2", "Othr", "Shed", "TenC", "NA"),
    "MiscVal": float,
    "MoSold": float,
    "YrSold": float,
    "SaleType": ("WD", "CWD", "VWD", "New", "COD", "Con", "ConLw", "ConLI", "ConLD", "Oth"),
    "SaleCondition": ("Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"),
    "SalePrice": float
}
_exclude_columns = ["Id"]
_label_name = "SalePrice"
_filename = "train.csv"
_train_filename = "data-train.csv"
_test_filename = "data-test.csv"
_model_filename = "GradesModel.joblib"

_ratio = 0.2
_seed = 42
_data_dir = "data/"
_out_dir = "out/"
_index_col = 0

@dataclass
class Globals:
    data: pd.DataFrame = pd.DataFrame()
    label_name: str = field(default_factory=str)
    filename: str = "data/train.csv"
    train_filename: str = "data/train.csv"
    test_filename: str = "data/test.csv"
    model_filename: str = "data/Data.joblib"
    columns: dict = field(default_factory=dict)
    exclude_columns: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    ratio: float = 0.2
    seed: int = 42
    data_dir: str = "data/"
    out_dir: str = "out/"
    index_col: int = 0

    def initialize(self):
        self.get_data()
        self.set_feature_names()
    
    def set_feature_names(self):
        if not self.feature_names:
            self.feature_names = list(self.columns.keys())
            if (self.label_name in self.feature_names):
                self.feature_names.remove(self.label_name)
            if (self.index_col > -1):
                self.feature_names.remove(self.feature_names[self.index_col])
        return self.feature_names
    
    def get_col_type(self, col_name):
        if isinstance(self.columns[col_name], tuple):
            return "categorical"
        else:
            return "numeric"
    
    def get_col_types(self):
        return {col: self.get_col_type(col) for col in self.feature_names}
    
    def get_categorical_features(self):
        return [col for col in self.feature_names if self.get_col_type(col) == "categorical"]
    
    def get_categorical_catagories(self):
        return [list(self.columns[col]) for col in self.feature_names if self.get_col_type(col) == "categorical"]
    
    def get_numeric_features(self):
        return [col for col in self.feature_names if self.get_col_type(col) == "numeric"]
    
    def get_data(self, filename=None, index_col=None):
        
        if filename is not None:
            fn = filename
        else:
            fn = self.filename
        
        if index_col is not None:
            ic = index_col
        else:
            ic = self.index_col
        
        self.data = pd.read_csv(fn, index_col=ic)
        return self.data
    
    def get_data_with_pandas_types(self, filename=None, index_col=None):
        """
        Assumes column 0 is the instance index stored in the
        csv file.  If no such column exists, remove the
        index_col=0 parameter.
        """
        
        if filename is not None:
            fn = filename
        else:
            fn = self.filename
        
        if index_col is not None:
            ic = index_col
        else:
            ic = self.index_col
        
        read_columns = self.columns.copy()
        
        for key in read_columns:
            if isinstance(read_columns[key], tuple):
                read_columns[key] = pd.CategoricalDtype(categories=read_columns[key])
            elif read_columns[key] == int:
                read_columns[key] = pd.Int64Dtype()
            elif read_columns[key] == float:
                read_columns[key] = pd.Float64Dtype()
            elif read_columns[key] == bool:
                read_columns[key] = pd.BooleanDtype()
            elif read_columns[key] == str:
                read_columns[key] = pd.StringDtype()
            else:
                logging.error(f"Unknown data type for column {key}")
                exit(1)
            
        if self.data.empty:
            try:
                if (ic >= 0):
                    self.data = pd.read_csv(fn, dtype=read_columns, index_col=ic)
                else:
                    self.data = pd.read_csv(fn, dtype=read_columns)
            except FileNotFoundError:
                logging.error(f"Data file not found: {fn}")
                exit(1)
        return self.data
            
def get_globals():
    glb = Globals(
        columns=_columns,
        exclude_columns=_exclude_columns,
        label_name=_label_name,
        filename= _data_dir + _filename,
        train_filename= _data_dir +_train_filename,
        test_filename= _data_dir +_test_filename,
        model_filename= _data_dir +_model_filename,
        ratio=_ratio,
        seed=_seed,
        data_dir=_data_dir,
        out_dir=_out_dir,
        index_col=_index_col
    )
    glb.initialize()
    
    # If the output directory does not exist, create it
    if not os.path.exists(glb.out_dir):
        os.makedirs(glb.out_dir)
    
    if (not os.path.exists(glb.out_dir + "/plots")):
        os.makedirs(glb.out_dir + "/plots")
    
    return glb

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True, glb=None):
        
        if glb is None:
            glb = get_globals()
        
        self.mCategoricalPredictors = glb.get_categorical_features()
        self.mNumericalPredictors = glb.get_numeric_features()
        self.mLabels = [glb.label_name]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        self.glb = glb
        
        if do_predictors:
            if do_numerical:
                self.mAttributes = self.mNumericalPredictors
            else:
                self.mAttributes = self.mCategoricalPredictors                
        else:
            self.mAttributes = self.mLabels
            
        return

    def fit( self, X, y=None ):
        # no fit necessary
        self.is_fitted_ = True
        return self

    def transform( self, X, y=None ):
        # only keep columns selected
        values = X[self.mAttributes]
        return values