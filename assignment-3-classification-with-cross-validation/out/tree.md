# tree

python3 pipeline.py grid-search -l loan_status -M tree -s 1 --categorical-missing-strategy most_frequent --numerical-missing-strategy mean

python3 pipeline.py show-best-params
    Best Score: 0.9281268676104028
    Best Params:
    {   'features__categorical__categorical-features-only__do_numerical': False,
        'features__categorical__categorical-features-only__do_predictors': True,
        'features__categorical__encode-category-bits__categories': 'auto',
        'features__categorical__encode-category-bits__handle_unknown': 'ignore',
        'features__categorical__missing-data__strategy': 'most_frequent',
        'features__numerical__missing-data__strategy': 'median',
        'features__numerical__numerical-features-only__do_numerical': True,
        'features__numerical__numerical-features-only__do_predictors': True,
        'model__criterion': 'entropy',
        'model__max_depth': 4,
        'model__max_features': None,
        'model__max_leaf_nodes': None,
        'model__min_impurity_decrease': 0.0,
        'model__min_samples_leaf': 1,
        'model__min_samples_split': 2,
        'model__splitter': 'best'
    }

python3 pipeline.py cross-score -l loan_status
    Cross Validation Score: 0.797 : ['0.839', '0.824', '0.728']

python3 pipeline.py score -l loan_status
    train: train_score: 0.9285531588370705

python3 pipeline.py loss -l loan_status
    train: L2(MSE) train_loss: 0.07144684116292949
    train: L1(MAE) train_loss: 0.07144684116292949
    train: R2 train_loss: 0.41489511284618363

python3 pipeline.py confusion-matrix -l loan_status
        t/p      F     T 
            F 47307.0 2988.0 
            T 5237.0 3113.0 
    Precision: 0.510
    Recall:    0.373
    F1:        0.431


python3 pipeline.py precision-recall-plot -l loan_status

python3 pipeline.py pr-curve -l loan_status