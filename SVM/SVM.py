import utils as u
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

os.chdir('./..')

# Given folds and a model(clf), apply LR on all folds and return the avg accuracy
def SVC_CV(folds, clf):
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
    clf = SVC()
    print(f'First base model -> Accuracy(CV): {SVC_CV(folds, clf):.4f}')

    # create a second version, tuning some hyperparameters
    clf = SVC(C=50, kernel='rbf', tol=0.5)
    print(f'Second version model -> Accuracy(CV): {SVC_CV(folds, clf):.4f}')

    # create a third version, main update here is the use of the StandardScaler
    clf = make_pipeline(StandardScaler(), SVC(C=50, kernel='rbf', tol=0.5))
    print(f'Third version model (StandardScaler) -> Accuracy(CV): {SVC_CV(folds, clf):.4f}')

    # final version, main update here is the change of the penalty from l2 to l1
    clf = make_pipeline(StandardScaler(), SVC(C=5, kernel='rbf', tol=0.5))
    print(f'Final model (StandardScaler with some changing in hyperparameters) -> Accuracy(CV): {SVC_CV(folds, clf):.4f}')

    return clf


# Testing the trained model on the test set (both passed as parameters)
def myTest(model, testSet):
    test_data, test_label = testSet
    predictions = model.predict(test_data)
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
    modelT = model.fit(train_data, train_label)  # trained model

    # We are now ready to test it on the training set
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

def myTestA():
    train_data, train_label = u.get_train()
    test_data, test_label = u.get_test()

    # clf = SVC(C=50, kernel='rbf', tol=0.5)
    clf = make_pipeline(StandardScaler(), SVC())

    clf.fit(train_data, train_label)
    print(clf.score(test_data, test_label))


if __name__ == '__main__':
    main()



