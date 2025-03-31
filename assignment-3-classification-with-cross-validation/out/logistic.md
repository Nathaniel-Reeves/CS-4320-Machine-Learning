# logistic

python3 pipeline.py grid-search -l loan_status -M logistic -s 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy mean
Fitting 3 folds for each of 2016 candidates, totalling 6048 fits

python3 pipeline.py show-best-params
    Best Score: 0.9121323323029674
    Best Params:
    {   'features__categorical__categorical-features-only__do_numerical': False,
        'features__categorical__categorical-features-only__do_predictors': True,
        'features__categorical__encode-category-bits__categories': 'auto',
        'features__categorical__encode-category-bits__handle_unknown': 'ignore',
        'features__categorical__missing-data__strategy': 'most_frequent',
        'features__numerical__missing-data__strategy': 'median',
        'features__numerical__numerical-features-only__do_numerical': True,
        'features__numerical__numerical-features-only__do_predictors': True,
        'model__C': 2.0,
        'model__class_weight': None,
        'model__dual': False,
        'model__fit_intercept': True,
        'model__intercept_scaling': 1,
        'model__max_iter': 1000,
        'model__n_jobs': -1,
        'model__penalty': 'l2',
        'model__random_state': 314159265,
        'model__solver': 'lbfgs',
        'model__tol': 0.0001,
        'model__verbose': 0,
        'model__warm_start': False
    }

python3 pipeline.py cross-score -l loan_status
    Cross Validation Score: 0.817 : ['0.823', '0.820', '0.808']

python3 pipeline.py score -l loan_status
    train: train_score: 0.9121493733481115

python3 pipeline.py loss -l loan_status
    train: L2(MSE) train_loss: 0.08785062665188847
    train: L1(MAE) train_loss: 0.08785062665188847
    train: R2 train_loss: 0.2805583821917752

python3 pipeline.py confusion-matrix -l loan_status
        t/p      F     T 
            F 44222.0 6073.0 
            T 3889.0 4461.0 
    Precision: 0.423
    Recall:    0.534
    F1:        0.472

python3 pipeline.py precision-recall-plot -l loan_status

python3 pipeline.py pr-curve -l loan_status