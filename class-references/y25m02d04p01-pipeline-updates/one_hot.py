#!/usr/bin/env python3

import sklearn.pipeline
import sklearn.preprocessing
import sklearn.base

import pandas as pd

class DataFrameSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, do_predictors=True, do_numerical=True):
        self.mCategoricalPredictors = ["RoofMatl"]
        self.mNumericalPredictors = ["BedroomAbvGr"]
        self.mLabels = ["SalePrice"]
        self.do_numerical = do_numerical
        self.do_predictors = do_predictors
        
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


filename = "data-train.csv"
data = pd.read_csv(filename, index_col=0)

items = []
items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
num_pipeline = sklearn.pipeline.Pipeline(items)


items = []
items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))

cat_pipeline = sklearn.pipeline.Pipeline(items)


#
# Merge data back together
#
items = []
items.append(("numerical", num_pipeline))
items.append(("categorical", cat_pipeline))
pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
#
#
#


pipeline.fit(data)
data_transform = pipeline.transform(data)
print(data_transform.shape)
# print(data_transform)
