# Random Forest

The technique is implemented in file `random_forest.py` using the scikit-learn `RandomForestClassifier`

There are 3 phases:
- Cross Validation for Hyperparameters tuning (in method `choose_hyper_rf()`)
- Training and Testing (in method `rand_forest()`)
- Evaluating models after dropping some attributes (in method `rand_forest()`)

### Hyperparameters tuning
Given the multitude of hyperparameters, I focused on the most important ones: criterion, max_features, 
n_estimators, and max_depth. After performing cross validation on the training set to fine tune these 
parameters the results are:

- criterion = entropy
- max_features = sqrt
- n_estimators = 500
- max_depth = 5

### Results
Metric scores:
```
[Full test dataset]        Accuracy: 0.7091, Precision: 0.7679, Recall: 0.5923, F-Score: 0.6688
[Dropping "SEX" column]    Accuracy: 0.7106, Precision: 0.7761, Recall: 0.5854, F-Score: 0.6674
[Dropping "PAY_0" column]  Accuracy: 0.6787, Precision: 0.7379, Recall: 0.5463, F-Score: 0.6278
```

Execution time (10000 training data, 3200 test data):
```
Training  time: 5650.777 ms
Inference time: 146.033 ms
```

Given that the accuracy for the trainig set is `0.7185` and for the test set is `0.7091`, no 
overfitting or underfitting is present.

