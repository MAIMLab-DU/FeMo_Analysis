# MLOps Reopository for MAIMLAB

### Legacy
```
RandomForestClassifier(class_weight={0: 1, 1: 1.2037037037037037},
                       min_samples_leaf=50, n_jobs=-1, random_state=0)
Train: [0.54166667 0.54166667 0.54166667 0.54166667 0.56521739]
Test: [0.56521739 0.56521739 0.56521739 0.56521739 0.48148148]
```

### Standard Grid Search
```
Result(best_model=RandomForestClassifier(class_weight={0: 1, 1: 1.2093023255813953},
                       min_samples_leaf=50, n_jobs=-1, random_state=0), best_model_hyperparams={'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': {0: 1, 1: 1.2093023255813953}, 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 50, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': -1, 'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False}, accuracy_scores={'train_accuracy': [0.45263157894736844, 0.45263157894736844, 0.45263157894736844, 0.45263157894736844, 0.4583333333333333], 'test_accuracy': [0.4583333333333333, 0.4583333333333333, 0.4583333333333333, 0.4583333333333333, 0.43478260869565216]})
```
### Optuna Search
```
Result(best_model=RandomForestClassifier(class_weight={0: 1, 1: 1.2093023255813953}, max_depth=10,
                       max_features=None, max_leaf_nodes=3, n_estimators=150,
                       random_state=0), best_model_hyperparams={'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': {0: 1, 1: 1.2093023255813953}, 'criterion': 'gini', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': 3, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'n_jobs': None, 'oob_score': False, 'random_state': 0, 'verbose': 0, 'warm_start': False}, accuracy_scores={'train_accuracy': [0.9157894736842105, 0.9473684210526315, 0.9263157894736842, 0.9578947368421052, 0.9479166666666666], 'test_accuracy': [0.7916666666666666, 0.8333333333333334, 0.9583333333333334, 0.8333333333333334, 0.782608695652174]})
```