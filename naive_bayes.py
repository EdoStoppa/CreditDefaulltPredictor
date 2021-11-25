from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sklearn import metrics

from utils import *

# Returns the accuracy as a result of the k-fold CV (used both by Gaussian and Bernoulli NB)
def naive_bayes_cv(nb, folds):
  accuracy = 0.0

  for train_data, train_labels, valid_data, valid_labels in folds:
    model = nb.fit(train_data, train_labels)
    accuracy += model.score(valid_data, valid_labels)

  return accuracy / len(folds) 

# Does the training and evaluate accuracy and other metrics on the test set
def test(nb, X_train, y_train, X_test, y_test):
  trained_model = nb.fit(X_train, y_train)
  y_pred = trained_model.predict(X_test)
  metrics = get_metrics(y_test, y_pred)
  return metrics


def main():

  # Cross validation on some hyperparameters
  folds = get_folds()

  # Baseline NB models
  gnb = GaussianNB()
  bnb = BernoulliNB()

  print(f"[GaussianNB - CV]  Average Accuracy with default parameters: {naive_bayes_cv(gnb, folds)}")
  print(f"[BernoulliNB - CV] Average Accuracy with default parameters: {naive_bayes_cv(bnb, folds)}")

  print("\n")

  # Now we try to tune hyperparameters for both models

  # We try to set the variance smoothing for Gaussian NB
  gnb = GaussianNB(var_smoothing=1)
  
  # And we try to change the binarization threshold on Bernoulli NB
  bnb = BernoulliNB(binarize=1.0)

  print(f"[GaussianNB - CV]  Average Accuracy with var_smoothing=1: {naive_bayes_cv(gnb, folds)}")
  print(f"[BernoulliNB - CV] Average Accuracy with binarize=1.0: {naive_bayes_cv(bnb, folds)}")

  print("\n")

  # Now we evaluate metrics on the test set

  X_train, y_train = get_train(drops=[])
  X_test, y_test = get_test(drops=[])

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score = test(gnb, X_train, y_train, X_test, y_test)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score = test(bnb, X_train, y_train, X_test, y_test)

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")

  # Now we try to drop some attributes to do some fairness considerations

  X_train, y_train = get_train(drops=["SEX"])
  X_test, y_test = get_test(drops=["SEX"])
  
  gnb = GaussianNB(var_smoothing=1)
  bnb = BernoulliNB(binarize=1.0)

  print("\nDropped the SEX column...\n")

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score = test(gnb, X_train, y_train, X_test, y_test)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score = test(bnb, X_train, y_train, X_test, y_test)

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")

  ###

  X_train, y_train = get_train(drops=["EDUCATION"])
  X_test, y_test = get_test(drops=["EDUCATION"])
  
  gnb = GaussianNB(var_smoothing=1)
  bnb = BernoulliNB(binarize=1.0)

  print("\nDropped the EDUCATION column...\n")

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score = test(gnb, X_train, y_train, X_test, y_test)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score = test(bnb, X_train, y_train, X_test, y_test)

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")


if __name__ == '__main__':
    main()  



