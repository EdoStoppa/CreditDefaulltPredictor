# Logistic Regression
###First part:
####(Tuning Hyperparameters)
The first phase is dedicated to tuning the **Hyperparameters** using the 
Cross Validation technique. This is done by the **myTraining()**, that return the best model found.

###Second Part:
####(Training and testing)
We are now ready to train the best model (found above), on the full training 
set.   
At this point, we are ready to test the trained model on the test set. We used 4 metrics 
to evaluate the model: **Accuracy, Precision, Recall, F-Score**.

###Third Part:
####(Dropping Sex/Pay column to see changes on the accuracy)
First, we try to drop the column **SEX**, to see if the accuracy change (should not).  
After, we try to drop the columns **PAY**, to show how they impact the accuracy of the model.