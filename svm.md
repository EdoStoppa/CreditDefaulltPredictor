# Support Vector Machines (Classifier)

The technique is implemented in file `svm.py` using the scikit-learn `SVC`

There are 3 phases:
- Cross Validation for Hyperparameters tuning
- Training and Testing 
- Evaluating models after dropping some attributes 

## Hyperparameters
- StandardScaler()
- C = 5
- kernel = 'rbf'
- tol = 0.5

## Results
```
Accuracy on training set: 0.7417

Test set evaluations: 
Accuracy: 0.7009, Precision: 0.7561, Recall: 0.5860, F-Score: 0.6603
```
Given that the accuracy for the trainig set and for the test set are really close, no overfitting or underfitting is present.

```
Time needed to train: 9785.46 ms
Time needed to do inference on test data: 4068.01 ms
```

The last thing that we are doing is to drop some attributes of the training dataset to see how the results change.
```
Test set evaluations, after dropping column "SEX":
Accuracy: 0.6984, Precision: 0.7537, Recall: 0.5822, F-Score: 0.6569

Test set evaluations, after dropping columns "PAY":
Accuracy: 0.6194, Precision: 0.5920, Recall: 0.7480, F-Score: 0.6609
```

Dropping the SEX column led to results that are very similar to those previously obtained.  

Conversely, dropping the PAY columns gave us worse results.
This is expected since these attributes are very important for the kind of prediction that we have to perform, so losing this information has a great influence on the achieved accuracy.