# Naive Bayes

Implemented techniques: Gaussian Naive Bayes and Bernoulli Naive Bayes.

3 phases:
- Cross Validation for Hyperparameters tuning
- Training and Testing
- Evaluating models after dropping some attributes

### Hyperparameters

- Gaussian NB: `var_smoothing` changed from default (1e-9) to 1.
- Bernoulli NB: `binarize` from default (0.0) to 1.0.

### Results

```
[GaussianNB - TEST]   Accuracy: 0.5509, Precision: 0.5305, Recall: 0.8217, F-Score: 0.6447
[BernoulliNB - TEST]  Accuracy: 0.6819, Precision: 0.7397, Recall: 0.5532, F-Score: 0.6330
```

Accuracy results are better for Bernoulli NB, probably because the input data is not following a Gaussian distribution, therefore in this case "binarizing" the input works better, and can be tuned with he `binarize` parameter as we did in our cross validation phase.

