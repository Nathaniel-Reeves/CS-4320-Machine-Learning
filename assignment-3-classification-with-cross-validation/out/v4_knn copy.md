# knn v4

python3 pipeline.py grid-search -l loan_status -M knn -s 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy mean
Fitting 3 folds for each of 30 candidates, totalling 90 fits

python3 pipeline.py show-best-params
    Best Score: 0.9368744181706736
    Best Params:
    {   'features__categorical__categorical-features-only__do_numerical': False,
        'features__categorical__categorical-features-only__do_predictors': True,
        'features__categorical__encode-category-bits__categories': 'auto',
        'features__categorical__encode-category-bits__handle_unknown': 'ignore',
        'features__categorical__missing-data__strategy': 'most_frequent',
        'features__numerical__missing-data__strategy': 'median',
        'features__numerical__numerical-features-only__do_numerical': True,
        'features__numerical__numerical-features-only__do_predictors': True,
        'model__algorithm': 'auto',
        'model__leaf_size': 1,
        'model__metric': 'minkowski',
        'model__n_jobs': -1,
        'model__n_neighbors': 20,
        'model__p': 1,
        'model__weights': 'distance'}

python3 pipeline.py cross-score -l loan_status
    Cross Validation Score: 0.850 : ['0.863', '0.861', '0.825']

python3 pipeline.py score -l loan_status
    train: train_score: 1.0

python3 pipeline.py loss -l loan_status
    train: L2(MSE) train_loss: 0.0
    train: L1(MAE) train_loss: 0.0
    train: R2 train_loss: 1.0

python3 pipeline.py confusion-matrix -l loan_status
        t/p      F     T 
            F 45288.0 5007.0 
            T 4081.0 4269.0 
    Precision: 0.460
    Recall:    0.511
    F1:        0.484



python3 pipeline.py precision-recall-plot -l loan_status

python3 pipeline.py pr-curve -l loan_status

python3 pipeline.py predict_proba
Kaggle Score Private: 0.9079
Kaggle Score Priate: 0.9145

