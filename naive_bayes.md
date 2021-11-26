# Naive Bayes

Implemented techniques: Gaussian Naive Bayes and Bernoulli Naive Bayes.

3 phases:
- Cross Validation for Hyperparameters tuning
- Training and Inference
- Evaluating models after dropping some attributes

### Hyperparameters tuning

- Gaussian NB: `var_smoothing` changed from default (1e-9) to 1.
- Bernoulli NB: `binarize` from default (0.0) to 1.0.

`var_smoothing` is an increase on the variance of the distribution.

`binarize` is the threshold used by BernoulliNB to convert the input data to binary features.

### Results

```
[GaussianNB - CV]  Average Accuracy with default parameters: 0.5418
[BernoulliNB - CV] Average Accuracy with default parameters: 0.6789
```

```
[GaussianNB - CV]  Average Accuracy with var_smoothing=1: 0.5531
[BernoulliNB - CV] Average Accuracy with binarize=1.0: 0.6828
```

Accuracy results are better for Bernoulli NB, probably because the input data is not following a Gaussian distribution, therefore in this case "binarizing" the input works better, and can be tuned with he `binarize` parameter as we did in our cross validation phase.

Then we train the model on the (balanced) training set and we did the inference both on training set and test set, to check if we have overfitting or underfitting:

```
[GaussianNB - TRAINING]   Accuracy: 0.5540, Precision: 0.5363, Recall: 0.8141, F-Score: 0.6466
[BernoulliNB - TRAINING]  Accuracy: 0.6839, Precision: 0.7372, Recall: 0.5741, F-Score: 0.6455


[GaussianNB - TEST]   Accuracy: 0.5509, Precision: 0.5305, Recall: 0.8217, F-Score: 0.6447
[BernoulliNB - TEST]  Accuracy: 0.6819, Precision: 0.7397, Recall: 0.5532, F-Score: 0.6330
```

As results are really close, we don't have underfitting or overfitting in this case.

Next, we evaluate the time needed to train the model and to do the inference on the test set:

```
[GaussianNB - TEST]  Time needed to train: 3.764 ms
[GaussianNB - TEST]  Time needed to do inference: 0.9 ms
[BernoulliNB - TEST] Time needed to train: 4.761 ms
[BernoulliNB - TEST] Time needed to do inference: 1.075 ms
```

The last thing that we are doing is to drop some attributes of the training dataset to see how the results change.

Dropping the `SEX` column led to results that are very similar to those previously obtained:

```
Dropped the SEX column...

[GaussianNB - TEST]   Accuracy: 0.5509, Precision: 0.5305, Recall: 0.8217, F-Score: 0.6447
[BernoulliNB - TEST]  Accuracy: 0.6787, Precision: 0.7375, Recall: 0.5469, F-Score: 0.6281
```

Conversely, dropping the `PAY` columns gave us worse results:

```
Dropped the PAY columns...

[GaussianNB - TEST]   Accuracy: 0.5509, Precision: 0.5305, Recall: 0.8217, F-Score: 0.6447
[BernoulliNB - TEST]  Accuracy: 0.5841, Precision: 0.6088, Recall: 0.4512, F-Score: 0.5183
```
This is expected since these attributes are very important for the kind of prediction that we have to perform, so losing this information has a great influence on the achieved accuracy.
