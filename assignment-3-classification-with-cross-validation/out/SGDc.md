# SGDc

python3 pipeline.py grid-search -l loan_status -M SGDc -s 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy mean
Fitting 3 folds for each of 180 candidates, totalling 540 fits

python3 pipeline.py show-best-params
Best Score: 0.9118424318738834
Best Params:
{   'features__categorical__categorical-features-only__do_numerical': False,
    'features__categorical__categorical-features-only__do_predictors': True,
    'features__categorical__encode-category-bits__categories': 'auto',
    'features__categorical__encode-category-bits__handle_unknown': 'ignore',
    'features__categorical__missing-data__strategy': 'most_frequent',
    'features__numerical__missing-data__strategy': 'median',
    'features__numerical__numerical-features-only__do_numerical': True,
    'features__numerical__numerical-features-only__do_predictors': True,
    'model__alpha': 0.001,
    'model__average': False,
    'model__class_weight': None,
    'model__early_stopping': False,
    'model__epsilon': 0.1,
    'model__eta0': 0.0,
    'model__fit_intercept': True,
    'model__l1_ratio': 0.15,
    'model__learning_rate': 'optimal',
    'model__loss': 'log_loss',
    'model__max_iter': 1000,
    'model__n_iter_no_change': 5,
    'model__n_jobs': -1,
    'model__penalty': 'l1',
    'model__power_t': 0.5,
    'model__random_state': 314159265,
    'model__shuffle': True,
    'model__tol': 0.001,
    'model__validation_fraction': 0.1,
    'model__warm_start': False}

python3 pipeline.py cross-score -l loan_status
    Cross Validation Score: 0.811 : ['0.869', '0.852', '0.711']

python3 pipeline.py score -l loan_status
    train: train_score: 0.9105124051496292

python3 pipeline.py loss -l loan_status
    train: L2(MSE) train_loss: 0.08948759485037087
    train: L1(MAE) train_loss: 0.08948759485037087
    train: R2 train_loss: 0.26715263776056597

python3 pipeline.py confusion-matrix -l loan_status
        t/p      F     T 
            F 40817.0 9478.0 
            T 3423.0 4927.0 
    Precision: 0.342
    Recall:    0.590
    F1:        0.433

python3 pipeline.py precision-recall-plot -l loan_status

python3 pipeline.py pr-curve -l loan_status