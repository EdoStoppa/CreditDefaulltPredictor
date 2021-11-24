import pandas as pd
from sklearn import metrics
import os

# Returns 2 numpy arrays of the full dataset: data, labels
def get_dataset_full(drops=[]):
    df = pd.read_csv(os.path.join('data', 'balanced_data.csv'))
    # Drop the selected columns
    if len(drops) > 0:
        for drop in drops:
            df = df.drop(drop, axis=1)
    # Take the column with the label
    labels = df['default_next_month'].to_numpy()
    # Remove the labels from the dataset
    data = df.drop('default_next_month', axis=1).to_numpy()

    return data, labels

# Returns 2 numpy arrays of the train dataset: data, labels
def get_train(drops=[]):
    df = pd.read_csv(os.path.join('data', 'train_balanced.csv'))
    # Drop the selected columns
    if len(drops) > 0:
        for drop in drops:
            df = df.drop(drop, axis=1)
    # Take the column with the label
    labels = df['default_next_month'].to_numpy()
    # Remove the labels from the dataset
    data = df.drop('default_next_month', axis=1).to_numpy()

    return data, labels

# Returns 2 numpy arrays of the test dataset: data, labels
def get_test(drops=[]):
    df = pd.read_csv(os.path.join('data', 'test_balanced.csv'))
    # Drop the selected columns
    if len(drops) > 0:
        for drop in drops:
            df = df.drop(drop, axis=1)
    # Take the column with the label
    labels = df['default_next_month'].to_numpy()
    # Remove the labels from the dataset
    data = df.drop('default_next_month', axis=1).to_numpy()

    return data, labels

# Split the dataset loaded from the specified csv file in k folds
# Returns of tuple composed by 4 elements:
#   1) train_data: numpy array of the data used to train the algorithm
#   2) train_labels: numpy array of the labels associated with train_data
#   3) valid_data: numpy array of the data used to validate the model
#   4) valid_labels: numpy array of the labels associated with valid_data
# Example use:
#   folds= get_folds()
#   for t_data, t_labels, v_data, v_labels in folds:
#       ...
def get_folds(k=5, csv=os.path.join('data', 'train_balanced.csv'), drops=[]):
    df = pd.read_csv(csv)
    # Drop the selected columns
    if len(drops) > 0:
        for drop in drops:
            df = df.drop(drop, axis=1)

    idx = k
    sets = []
    # Split the dataset in k partition
    while idx > 0:
        sets.append(df.sample(frac=(1/idx), random_state=1))
        df = df.drop(sets[-1].index)
        idx -= 1

    cross_val_sets = []
    for idx in range(len(sets)):
        # For each partition, get the the validation and train dataframe
        validation = sets[idx]
        sets_copy = sets.copy()
        del sets_copy[idx]
        train = pd.concat(sets_copy)

        # Dive each set in data and labels, then convert everything to numpy arrays
        valid_labels = validation['default_next_month'].to_numpy()
        valid_data = validation.drop('default_next_month', axis=1).to_numpy()
        train_labels = train['default_next_month'].to_numpy()
        train_data = train.drop('default_next_month', axis=1).to_numpy()

        # Append to list the 4 elements
        cross_val_sets.append((train_data, train_labels, valid_data, valid_labels))

    return cross_val_sets

# Takes the ground truths and the predictions, then return 4 metrics:
def get_metrics(truths, predictions):
    accuracy = metrics.accuracy_score(truths, predictions)
    precision = metrics.precision_score(truths, predictions)
    recall = metrics.recall_score(truths, predictions)
    f_score = metrics.f1_score(truths, predictions)

    return accuracy, precision, recall, f_score

def save_balanced_data():
    df = pd.read_csv(os.path.join('data', 'original_data.csv'))

    # Divide dataset in the 2 classes
    df_0 = df[df["default_next_month"] == 0]
    df_1 = df[df["default_next_month"] == 1]

    # Shuffle them
    df_0 = df_0.sample(frac=1).reset_index(drop=True)
    df_1 = df_1.sample(frac=1).reset_index(drop=True)

    # Get same number of sample
    df_0 = df_0.head(6600)
    df_1 = df_1.head(6600)

    # Concatenate the 2 sub-datasets
    df = pd.concat([df_0, df_1])
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Divide in train and test dataset
    test = df.head(3200)
    train = df.drop(test.index)

    # Save everything to a csv
    df.to_csv(os.path.join('data', 'balanced_data.csv'), index=False)
    train.to_csv(os.path.join('data', 'train_balanced.csv'), index=False)
    test.to_csv(os.path.join('data', 'test_balanced.csv'), index=False)

# Used only for testing
if __name__ == '__main__':
    #get_dataset_full()
    #get_folds()

    save_balanced_data()