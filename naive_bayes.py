from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from datetime import datetime
from utils import *

# Returns the accuracy as a result of the k-fold CV (used both by Gaussian and Bernoulli NB)
def naive_bayes_cv(nb, folds):
  accuracy = 0.0

  for train_data, train_labels, valid_data, valid_labels in folds:
    model = nb.fit(train_data, train_labels)
    accuracy += model.score(valid_data, valid_labels)

  return accuracy / len(folds) 

# Does the training and evaluate accuracy and other metrics on the test set
def test(nb, X_train, y_train, X_test, y_test, return_times = False):
  start_time_fit = datetime.now()
  trained_model = nb.fit(X_train, y_train)
  end_time_fit = datetime.now()

  start_time_predict = datetime.now()
  y_pred = trained_model.predict(X_test)
  end_time_predict = datetime.now()

  metrics = get_metrics(y_test, y_pred) 
  
  if(return_times):
    metrics = metrics + ((end_time_fit - start_time_fit).total_seconds() * 1000, (end_time_predict - start_time_predict).total_seconds() * 1000)

  return metrics


def main():

  ### Cross validation on some hyperparameters

  folds = get_folds()

  # Baseline NB models
  gnb = GaussianNB()
  bnb = BernoulliNB()

  print(f"[GaussianNB - CV]  Average Accuracy with default parameters: {naive_bayes_cv(gnb, folds)}")
  print(f"[BernoulliNB - CV] Average Accuracy with default parameters: {naive_bayes_cv(bnb, folds)}")

  print("\n")

  ### Now we try to tune hyperparameters for both models

  # We try to set the variance smoothing for Gaussian NB
  gnb = GaussianNB(var_smoothing=1)
  
  # And we try to change the binarization threshold on Bernoulli NB
  bnb = BernoulliNB(binarize=1.0)

  print(f"[GaussianNB - CV]  Average Accuracy with var_smoothing=1: {naive_bayes_cv(gnb, folds)}")
  print(f"[BernoulliNB - CV] Average Accuracy with binarize=1.0: {naive_bayes_cv(bnb, folds)}")

  print("\n")

  ### Now we evaluate metrics on the training set and test set, evaluating the accuracy on the latter and the possible under/over fitting

  X_train, y_train = get_train()
  X_test, y_test = get_test()

  gnb_accuracy_train, gnb_precision_train, gnb_recall_train, gnb_f_score_train = test(gnb, X_train, y_train, X_train, y_train)
  bnb_accuracy_train, bnb_precision_train, bnb_recall_train, bnb_f_score_train = test(bnb, X_train, y_train, X_train, y_train)

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score, gnb_time_to_fit, gnb_time_to_predict = test(gnb, X_train, y_train, X_test, y_test, return_times=True)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score, bnb_time_to_fit, bnb_time_to_predict = test(bnb, X_train, y_train, X_test, y_test, return_times=True)

  print(f"[GaussianNB - TRAINING]   Accuracy: {gnb_accuracy_train:.4f}, Precision: {gnb_precision_train:.4f}, Recall: {gnb_recall_train:.4f}, F-Score: {gnb_f_score_train:.4f}")
  print(f"[BernoulliNB - TRAINING]  Accuracy: {bnb_accuracy_train:.4f}, Precision: {bnb_precision_train:.4f}, Recall: {bnb_recall_train:.4f}, F-Score: {bnb_f_score_train:.4f}")

  print("\n")

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")

  print(f"[GaussianNB - TEST] Time needed to train: {gnb_time_to_fit} ms")
  print(f"[GaussianNB - TEST] Time needed to do inference: {gnb_time_to_predict} ms")

  print(f"[BernoulliNB - TEST] Time needed to train: {bnb_time_to_fit} ms")
  print(f"[BernoulliNB - TEST] Time needed to do inference: {bnb_time_to_predict} ms")

  ### Now we try to drop some attributes to do some considerations about bias

  X_train, y_train = get_train(drops=["SEX"])
  X_test, y_test = get_test(drops=["SEX"])
  
  gnb = GaussianNB(var_smoothing=1)
  bnb = BernoulliNB(binarize=1.0)

  print("\nDropped the SEX column...\n")

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score = test(gnb, X_train, y_train, X_test, y_test)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score = test(bnb, X_train, y_train, X_test, y_test)

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")

  X_train, y_train = get_train(drops=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
  X_test, y_test = get_test(drops=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
  
  gnb = GaussianNB(var_smoothing=1)
  bnb = BernoulliNB(binarize=1.0)

  print("\nDropped the PAY columns...\n")

  gnb_accuracy, gnb_precision, gnb_recall, gnb_f_score = test(gnb, X_train, y_train, X_test, y_test)
  bnb_accuracy, bnb_precision, bnb_recall, bnb_f_score = test(bnb, X_train, y_train, X_test, y_test)

  print(f"[GaussianNB - TEST]   Accuracy: {gnb_accuracy:.4f}, Precision: {gnb_precision:.4f}, Recall: {gnb_recall:.4f}, F-Score: {gnb_f_score:.4f}")
  print(f"[BernoulliNB - TEST]  Accuracy: {bnb_accuracy:.4f}, Precision: {bnb_precision:.4f}, Recall: {bnb_recall:.4f}, F-Score: {bnb_f_score:.4f}")

def default_accuracy():
    clf = BernoulliNB()
    train_data, train_label = get_train()
    test_data, test_label = get_test()
    clf.fit(train_data, train_label)
    print(f'Accuracy on training set: {clf.score(train_data, train_label)}')
    print(f'Accuracy on test set: {clf.score(test_data, test_label)}')

if __name__ == '__main__':
  main()
  #default_accuracy()



