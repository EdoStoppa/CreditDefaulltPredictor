#Credit Default Predictor
This project aimed to classify the defaulted payments of customers using five ML techniques.  
The dataset contains 30000 instances of customers in Taiwan (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

### The techniques used are:
- Logistic Regression : [md](logistic_regression.md) | [code](logistic_regression.py)
- SVM (Classifier) : [md](svm.md) | [code](svm.py)
- Gaussian Naive Bayes : [md](naive_bayes.md) | [code](naive_bayes.py)
- Bernoulli Naive Bayes : [md](naive_bayes.md) | [code](naive_bayes.py)
- Random Forest : [md](random_forest.md) | [code](random_forest.py)

## Preprocessing data
To be done

## Method

### First phase:
#### (Tuning Hyperparameters)
The first phase is dedicated to tuning the **Hyperparameters** using the
Cross Validation technique. 

### Second phase:
#### (Training and testing)
We are now ready to train the best model (found above), on the full training
set.
At this point, we are ready to test the trained model on the test set. We used 4 metrics
to evaluate the model: **Accuracy, Precision, Recall, F-Score**.

### Third phase:
#### (Dropping Sex/Pay column to see changes on the accuracy)
First, we try to drop the column **SEX**, to see if the accuracy change (should not).
After, we try to drop the columns **PAY**, to show how they impact the accuracy of the model.
