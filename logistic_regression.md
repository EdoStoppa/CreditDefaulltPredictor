# Logistic Regression
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
- solver = 'liblinear'
- penalty = 'l1'
- C = 0.4
- intercept_scaling = 1e-4
- class_weight = 'balanced'

## Results
```
Test set evaluations: 
Accuracy: 0.6781, Precision: 0.6909, Recall: 0.6352, F-Score: 0.6619

Test set evaluations, after dropping column "SEX":
Accuracy: 0.6731, Precision: 0.6857, Recall: 0.6295, F-Score: 0.6564

Test set evaluations, after dropping columns "PAY":
Accuracy: 0.6194, Precision: 0.5962, Recall: 0.7202, F-Score: 0.6524
```