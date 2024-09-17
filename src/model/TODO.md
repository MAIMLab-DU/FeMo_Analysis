### Model selection pipeline

```
[ init ] → [ search ] → [ fit ]
```

## Implement tuning with Optuna

**Update fit() method, same classifier is now being trained every fold. Should be different classifier?**

- [x] `LogisticRegressionClassifier` -> **search** -> **fit** test passed
  - [x] Search with *Optuna* test passed
- [x] `SVMClassifier` -> **search** -> **fit** test passed
  - [x] Search with *Optuna* test passed
- [x] `NeuralNetClassifier` -> **search** -> **fit** test passed
  - [x] Search with *Optuna* test passed
- [x] `RandomForestClassifier` -> **search** -> **fit** test passed
  - [x] Search with *Optuna* test passed
- [x] `AdaboostClassifier` -> **search** -> **fit** test passed
  - [x] Search with *Optuna* test passed
- [ ] `EnsembleClassifier` -> **search** -> **fit** test passed
  - [ ] Search with *Optuna* test passed