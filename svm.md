# Support Vector Machines (Classifier)
### First part:
#### (Tuning Hyperparameters)
The first phase is dedicated to tuning the **Hyperparameters** using the 
Cross Validation technique. This is done by the **myTraining()**, that return the best model found.

### Second Part:
#### (Training and testing)
We are now ready to train the best model (found above), on the full training 
set.   
At this point, we are ready to test the trained model on the test set. We used 4 metrics 
to evaluate the model: **Accuracy, Precision, Recall, F-Score**.

### Third Part:
#### (Dropping Sex/Pay column to see changes on the accuracy)
First, we try to drop the column **SEX**, to see if the accuracy change (should not).  
After, we try to drop the columns **PAY**, to show how they impact the accuracy of the model.

## Hyperparameters
- StandardScaler()
- C = 5
- kernel = 'rbf'
- tol = 0.5

## Results
```
Test set evaluations: 
Accuracy: 0.7009, Precision: 0.7561, Recall: 0.5860, F-Score: 0.6603

Test set evaluations, after dropping column "SEX":
Accuracy: 0.6984, Precision: 0.7537, Recall: 0.5822, F-Score: 0.6569

Test set evaluations, after dropping columns "PAY":
Accuracy: 0.6194, Precision: 0.5920, Recall: 0.7480, F-Score: 0.6609
```