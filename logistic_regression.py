from sklearn.linear_model import LogisticRegression
import utils as u
from datetime import datetime


# Given folds and a model(clf), apply LR on all folds and return the avg accuracy
def logistic_regression_CV(folds, clf):
    accuracy = 0
    for train_data, train_label, validation_data, validation_label in folds:
        clf.fit(train_data, train_label)
        accuracy += clf.score(validation_data, validation_label)
    return accuracy / len(folds)


# Tuning Hyperparameters based on training and testing on CV folds. Return the best model
def myTraining():
    # Get folds from utils.py
    folds = u.get_folds()

    # create a first base model, try it applying CV and print the avg Accuracy
    clf = LogisticRegression(max_iter=1000)
    print(f'First base model -> Accuracy(CV): {logistic_regression_CV(folds, clf):.4f}')

    # create a second version, tuning some hyperparameters
    clf = LogisticRegression(C=0.4, intercept_scaling=1e-4, class_weight='balanced', max_iter=1000)
    print(f'Second version model -> Accuracy(CV): {logistic_regression_CV(folds, clf):.4f}')

    # create a third version, main update here is the change of the solver
    clf = LogisticRegression(solver='liblinear', penalty='l2', C=0.4, intercept_scaling=1e-4, class_weight='balanced',
                             max_iter=1000)
    print(f'Third version model (solver: liblinear) -> Accuracy(CV): {logistic_regression_CV(folds, clf):.4f}')

    # final version, main update here is the change of the penalty from l2 to l1
    clf = LogisticRegression(solver='liblinear', penalty='l1', C=0.4, intercept_scaling=1e-4, class_weight='balanced',
                             max_iter=1000)
    print(f'\nFinal model (solver: liblinear, penalty: l1) -> Accuracy(CV): {logistic_regression_CV(folds, clf):.4f}')

    return clf


# Testing the trained model on the test set (both passed as parameters)
def myTest(model, testSet):
    test_data, test_label = testSet

    start_time_inf = datetime.now()
    predictions = model.predict(test_data)
    end_time_inf = datetime.now()
    print(f"Time needed to do inference on test data: " +
          f"{(end_time_inf - start_time_inf).total_seconds() * 1000:.4f} ms")
    # Using the get_metrics function provided in utils.py to get multiple evaluations
    accuracy, precision, recall, f_score = u.get_metrics(test_label, predictions)
    print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F-Score: {f_score:.4f}')


def main():

    #FIRST PART

    # Tuning hyperparameters based on CV testing, to get the best model
    model = myTraining()


    #SECOND PART

    '''After tuning the Hyperparameters and finding the best configuration based on the avg Accuracy given by CV,
       we train the model on the full training set'''
    train_data, train_label = u.get_train()

    start_time_fit = datetime.now()
    modelT = model.fit(train_data, train_label)  # trained model
    end_time_fit = datetime.now()
    print(f"Time needed to train: " +
          f"{(end_time_fit - start_time_fit).total_seconds() * 1000:.2f} ms")

    #Evaluations on training set
    print(f'Accuracy on training set: {modelT.score(train_data, train_label):.4f}')

    # We are now ready to test it on the test set
    testSet = u.get_test()
    print("\nTest set evaluations: ")
    myTest(modelT, testSet)


    #THIRD PART

    # At this point we try to drop the column SEX, test the model on these new set, to see if the accuracy change (should not)
    train_data, train_label = u.get_train(drops=["SEX"])
    modelT = model.fit(train_data, train_label)
    testSet = u.get_test(drops=["SEX"])
    print("\nTest set evaluations, after dropping column \"SEX\":")
    myTest(modelT, testSet)


    # Finally, we try to drop the "PAY" columns, to show how they impact the accuracy of the model
    train_data, train_label = u.get_train(drops=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
    modelT = model.fit(train_data, train_label)
    testSet = u.get_test(drops=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"])
    print("\nTest set evaluations, after dropping columns \"PAY\":")
    myTest(modelT, testSet)

def default_accuracy():
    clf = LogisticRegression(max_iter=1000)
    train_data, train_label = u.get_train()
    test_data, test_label = u.get_test()
    clf.fit(train_data, train_label)
    print(f'Accuracy on training set: {clf.score(train_data, train_label)}')
    print(f'Accuracy on test set: {clf.score(test_data, test_label)}')


if __name__ == '__main__':
    main()
    #default_accuracy()
