# Logistic Regression

The technique is implemented in file `logistic_regression.py` using the scikit-learn `LogisticRegression`

There are 3 phases:
- Cross Validation for Hyperparameters tuning
- Training and Testing 
- Evaluating models after dropping some attributes 


## Hyperparameters
- solver = liblinear
- penalty = l1
- C = 0.4
- intercept_scaling = 1e-4
- class_weight = balanced

## Results
```
Accuracy on training set: 0.6756

Test set evaluations: 
Accuracy: 0.6781, Precision: 0.6909, Recall: 0.6352, F-Score: 0.6619
```
Given that the accuracy for the trainig set and for the test set are really close, no overfitting or underfitting is present.  

```
Time needed to train: 320.82 ms
Time needed to do inference on test data: 0.9990 ms
```
The last thing that we are doing is to drop some attributes of the training dataset to see how the results change.
```
Test set evaluations, after dropping column "SEX":
Accuracy: 0.6731, Precision: 0.6857, Recall: 0.6295, F-Score: 0.6564

Test set evaluations, after dropping columns "PAY":
Accuracy: 0.6194, Precision: 0.5962, Recall: 0.7202, F-Score: 0.6524
```

Dropping the SEX column led to results that are very similar to those previously obtained.  
Conversely, dropping the PAY columns gave us worse results.
This is expected since these attributes are very important for the kind of prediction that we have to perform, so losing this information has a great influence on the achieved accuracy.