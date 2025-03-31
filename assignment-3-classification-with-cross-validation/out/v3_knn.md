# knn v2

python3 pipeline.py grid-search -l loan_status -M knn -s 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy mean
Fitting 3 folds for each of 90 candidates, totalling 270 fits

python3 pipeline.py show-best-params
    Best Score: 0.9372836698155228
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
        'model__n_neighbors': 10,
        'model__p': 1,
        'model__weights': 'distance'
    }

python3 pipeline.py cross-score -l loan_status
    Cross Validation Score: 0.834 : ['0.869', '0.814', '0.820']

python3 pipeline.py score -l loan_status
    train: train_score: 1.0

python3 pipeline.py loss -l loan_status
    train: L2(MSE) train_loss: 0.0
    train: L1(MAE) train_loss: 0.0
    train: R2 train_loss: 1.0

python3 pipeline.py confusion-matrix -l loan_status
            t/p      F     T 
                F 45891.0 4404.0 
                T 4510.0 3840.0 
        Precision: 0.466
        Recall:    0.460
        F1:        0.463


python3 pipeline.py precision-recall-plot -l loan_status

python3 pipeline.py pr-curve -l loan_status

python3 pipeline.py predict
Kaggle Score Private: 0.81776
Kaggle Score Priate: 0.82940

