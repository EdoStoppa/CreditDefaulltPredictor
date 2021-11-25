import numpy as np
import utils as ut
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
import warnings
from datetime import datetime

def validate(data_list, model, name, verbose=False):
    acc = []
    for train_data, train_labels, test_data, test_labels in data_list:
        # Train
        model.fit(train_data, train_labels)
        # Obtain the predictions
        predictions = model.predict(test_data)
        # Test the model
        accuracy, b, c, d = ut.get_metrics(test_labels, predictions)
        # Record accuracy
        acc.append(accuracy)
    # Calculate the mean accuracy
    mean = np.mean(acc)
    if verbose: print(f'{name} Train set accuracy: {mean * 100:3f}')
    return mean

# After running this function, the results are:
#       "The best parameters are: criterion=entropy, max_features=sqrt, n_estimators=500, max_depth=5"
def choose_hyper_rf(cross_val_list, verbose=False):
    name = 'Random Forest'
    print(f'\nStarted {name} workload...')

    # Cross validate criterion and Max Features
    models = list()
    models.append(RandomForestClassifier(criterion='gini', max_features='sqrt'))
    models.append(RandomForestClassifier(criterion='entropy', max_features='sqrt'))
    models.append(RandomForestClassifier(criterion='gini', max_features='log2'))
    models.append(RandomForestClassifier(criterion='entropy', max_features='log2'))

    idx_models = list()
    for i in range(10):
        accuracies = []
        for idx in range(len(models)):
            model = models[idx]
            name_ = f'{name} {idx}'
            acc = validate(cross_val_list, model, name_, verbose=verbose)
            accuracies.append(acc)

        winner_idx = np.argmax(accuracies)
        idx_models.append(winner_idx)

        print(f'The winner is model {winner_idx}')

    final_idxs = np.bincount(idx_models)
    winner_idx = np.argmax(final_idxs)
    if   winner_idx == 0: criterion, max_features = 'gini', 'sqrt'
    elif winner_idx == 1: criterion, max_features = 'entropy', 'sqrt'
    elif winner_idx == 2: criterion, max_features = 'gini', 'log2'
    elif winner_idx == 3: criterion, max_features = 'entropy', 'log2'
    else: criterion, max_features = 'gini', 'sqrt'

    winner_params = [criterion, max_features]

    # Cross Validate number of estimators
    models = list()
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=100))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=200))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=300))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=400))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features, n_estimators=500))

    idx_models = list()
    for i in range(10):
        accuracies = []
        for idx in range(len(models)):
            model = models[idx]
            name_ = f'{name} {idx}'
            acc = validate(cross_val_list, model, name_, verbose=verbose)
            accuracies.append(acc)

        winner_idx = np.argmax(accuracies)
        idx_models.append(winner_idx)

        print(f'The winner is model {winner_idx}')

    final_idxs = np.bincount(idx_models)
    winner_idx = np.argmax(final_idxs)
    if   winner_idx == 0: n_estimators = 100
    elif winner_idx == 1: n_estimators = 200
    elif winner_idx == 2: n_estimators = 300
    elif winner_idx == 3: n_estimators = 400
    elif winner_idx == 4: n_estimators = 500
    else:                 n_estimators = 100

    # Cross validate max tree depth
    models = list()
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=3))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=5))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=7))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=9))
    models.append(RandomForestClassifier(criterion=criterion, max_features=max_features,
                                         n_estimators=n_estimators, max_depth=11))

    idx_models = list()
    for i in range(10):
        accuracies = []
        for idx in range(len(models)):
            model = models[idx]
            name_ = f'{name} {idx}'
            acc = validate(cross_val_list, model, name_, verbose=verbose)
            accuracies.append(acc)

        winner_idx = np.argmax(accuracies)
        idx_models.append(winner_idx)

        print(f'The winner is model {winner_idx}\n')

    final_idxs = np.bincount(idx_models)
    winner_idx = np.argmax(final_idxs)
    if winner_idx == 0:   max_depth = 3
    elif winner_idx == 1: max_depth = 5
    elif winner_idx == 2: max_depth = 7
    elif winner_idx == 3: max_depth = 9
    elif winner_idx == 4: max_depth = 11
    else:                 max_depth = None

    winner_params += [max_depth]

    print(f'The best parameters are:\n' +
          f'    criterion={winner_params[0]}, ' +
          f'    max_features={winner_params[1]}, ' +
          f'    n_estimators={winner_params[2]}' +
          f'    max_depth={winner_params[3]}')

    return RandomForestClassifier(criterion=winner_params[0],
                                  max_features=winner_params[1],
                                  n_estimators=winner_params[2],
                                  max_depth=winner_params[3])

def complete_metrics_random_forest(train_, test_, cross_model=None):
    train_data, train_labels = train_
    test_data, test_labels = test_
    name = 'Random Forest Classifier'

    if cross_model is None:
        model = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=500, max_depth=5)
    else:
        model = cross_model

    accs, precs, recs, fs = [], [], [], []
    for i in range(10):
        model.fit(train_data, train_labels)
        # Obtain the predictions
        predictions = model.predict(test_data)
        # Test the model
        accuracy, precision, recall, f_score = ut.get_metrics(test_labels, predictions)
        accs.append(accuracy), precs.append(precision), recs.append(recall), fs.append(f_score)

    accuracy, precision, recall, f_score = np.mean(accs), np.mean(precs), np.mean(recs), np.mean(fs)
    print(f'\nFinal Scores: -> Accuracy: {accuracy}, Precision: {precision}, ' +
          f'Recall: {recall}, F-Score: {f_score}')

    return accuracy, precision, recall, f_score

def check_stats(train_, test_, model, space=''):
    train_data, train_labels = train_
    test_data, test_labels = test_

    start_time_fit = datetime.now()
    model.fit(train_data, train_labels)
    end_time_fit = datetime.now()
    print(f"{space}Time needed to train: " +
          f"{(end_time_fit - start_time_fit).total_seconds() * 1000} ms")

    start_time_inf = datetime.now()
    predictions_test = model.predict(test_data)
    end_time_inf = datetime.now()
    print(f"{space}Time needed to do inference on test data: " +
          f"{(end_time_inf - start_time_inf).total_seconds() * 1000} ms")
    predictions_train = model.predict(train_data)

    a, b, c, d = ut.get_metrics(train_labels, predictions_train)
    print_scores(a, b, c, d, 'Train')
    a, b, c, d = ut.get_metrics(test_labels, predictions_test)
    print_scores(a, b, c, d, 'Test')

def print_scores(a, b, c, d, typ):
    print(f'    Final {typ} Scores: -> Accuracy: {a}, Precision: {b}, ' +
          f'Recall: {c}, F-Score: {d}')

def rand_forest():
    print('\nRandom Forest workload\n')
    base_model = RandomForestClassifier()
    second_model = RandomForestClassifier(criterion='entropy', n_estimators=300)
    cross_model = RandomForestClassifier(criterion='entropy', max_features='sqrt', n_estimators=500, max_depth=5)

    train_data = ut.get_train()
    test_data  = ut.get_test()

    train_data_no_s = ut.get_train(drops=['SEX'])
    test_data_no_s = ut.get_test(drops=['SEX'])

    train_data_no_p = ut.get_train(drops=['PAY_0'])
    test_data_no_p = ut.get_test(drops=['PAY_0'])

    names = ['Default Model', 'Enhanced Model', 'Final Model']
    models = [base_model, second_model, cross_model]
    for idx in range(len(names)):
        print(f'{names[idx]}')
        print('    Full data...')
        check_stats(train_data, test_data, models[idx], space='    ')
        print('\n    Without column "SEX"...')
        check_stats(train_data_no_s, test_data_no_s, models[idx], space='    ')
        print('\n    Without column "PAY_0"...')
        check_stats(train_data_no_p, test_data_no_p, models[idx], space='    ')
        print('\n')

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    rand_forest()