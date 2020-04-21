---
title: "Entry 26: Setting thresholds - precision, recall, and ROC"
categories:
  - Blog
tags:
  - model-eval
---

I don't always want the default threshold for determining the classification (negative or positive) the way I did in <font color='red'>Entry 24</font>. As discussed in the precision / recall tradeoff section of <font color='red'>Entry 23</font> sometimes there will be a better threshold.

## The Problem

*[Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)* points out that not all models provide a realistic representation of uncertainity. If overfitted, a random forest model will be 100% certain for every prediction, even if it's almost never right.

I need a way to determine the best threshold for my purposes for the specific model I'm looking at and a way to compare various models to determine which one is most appropriate for the use case.

## The Options

### Precision and recall vs thresholds

Precision and recall are plotted as their own lines on the y-axis against threshold on the x-axis in [Hands-On Machine Learning with Scikit-Learn & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291).

This is the view I used in <font color='red'>Entry 25</font> to illustrate the precision / recall tradeoff. It's very good for visualizing the tradeoff between precision and recall and where the intersection of that tradeoff lies.

The best illustration of it on the smaller real world datasets I ran is in <font color='red'>Entry 25c notebook - Titanic</font>.

I also created a notebook that used a lot of code from *Hands-On Machine Learning with Scikit-Learn & TensorFlow* in <font color='red'>Entry 25e notebook - MNIST</font>. I did this to illustrate the difference between the PR AUC and ROC AUC, but it is a better example of percision and recall vs thresholds as well. It has significantly more data, so the lines are smoother. 

### Precision-recall (PR) curve

Precision is plotted on the y-axis with recall on the x-axis. The pretty, theoritical, lots-of-data-behind-it line is a logarithmic decay where percision starts at 1 in the upper left corner and ends with percision at 0 in the lower right. An example is in <font color='red'>Entry 25e notebook - MNIST</font>.

In the smaller datasets the lines are messy, bumpy, and don't start or stop exactly at 1 or 0.

### Precision-recall area under the curve (PR AUC)

The area under the curve (AUC) is exactly what it sounds like. The PR curve divides the chart into two sides. The closer the curve is to the upper left the more space will be under that curve. AUC calculates the area that lies underneath the curve to provide a general metric for how well the model performs.

PR AUC is the area under the percision / recall curve.

### Receiver operating characteristic (ROC) curve

This plots the true positive rate (also known as recall and sensitivity) on the y-axis and the false positive rate on the x-axis (false positive rate can be represented mathematically as $1 - sensitivity$).  This provides the sensitivity vs specificity comparison that [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0) recommends.

As long as the model predicts better than random guessing, the plot of ROC is a logarithmic growth curve.  Per *Introduction to Machine Learning with Python*:

> Predicting randomly always produces an AUC of 0.5, no matter how imbalanced the classes in a dataset are. This makes AUC a much better metric for imbalanced classification problems than accuracy.

The better the model, the closer the curve will be to the upper left hand corner of the plot. Random guessing results in a straight diagonal line that runs from the lower left to the upper right. I included the random guessing line in the  ROC charts of all five of the notebooks.

### ROC AUC

The AUC portion of the ROC AUC work the same as the PR AUC - area under the curve literally measures what percentage of the chart falls under the curve. The difference between PR AUC and ROC AUC is the curve that we use to specify the divinding line is the ROC curve instead of the PR curve. It's really that easy.

## The Proposed Solution

### ROC AUC vs PR AUC

*Hands-On Machine Learning with Scikit-Learn & TensorFlow* advises:

> As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives, and the ROC curve otherwise.

*Applied Predictive Modeling* states:

> One advantage of using ROC curves to characterize models is that, since it is a function of sensitivity and specificity, the curve is insensitive to disparities in the class proportions (Provost et al. 1998; Fawcett 2006).

Then of course there's also the quote above from *Introduction to Machine Learning with Python*.

I discussed this conflicting information with my coworker [Sabber](https://medium.com/@sabber). He agreed that both metrics are good for imbalanced classes, but that the best choice for imbalanced classes comes down to the definitions of the underlying metrics.

The PR curve is based on precision and recall. As a reminder, *precision* is the rate of correct positive predictions out of all positive predictions (based on prediction population) and *recall* is the rate of correct positive predictions out of all positive observations (based on observed population).

$precision = \frac{TP}{TP+FP} = \frac{TP}{PP}$

$recall = \frac{TP}{TP + FN} = \frac{TP}{AP}$


The ROC curve is based on sensitivity and specifity. As a reminder, *sensitivity* is another name for recall and *specifity* is the rate of correct negative predictions out of all negative observations (based on observed population).

$specificity = \frac{TN}{TN + FP} = \frac{TN}{AN}$

Based on the recap above, it's easy to see that the PR curve considers true positive (TP), false positive (FP), and false negative (FN), whereas the ROC curve considers all four: true positive (TP), false positive (FP), true negative (TN), and false negative (FN).

Because the ROC curve includes true negatives, this means that a majority class can skew the results if the model is good at identifying the majority class.

*Hands-On Machine Learning with Scikit-Learn & TensorFlow* had a nice example of this using the [MNIST digits dataset](https://www.openml.org/d/554). He predicted whether a handwriten digit was a `5`, which turned the dataset into a binary classification problem.

I replicated his example using the pipeline I've been developing and the same dataset as Aurelien Geron. The difference is subtle but noticable when looking at the charts. The ROC AUC would return a higher evaluation of the mode's performance than a PR AUC would

### Decision

I ran all of the metrics and made graphs for all the options listed above. Ultimately, to compare models I believe I'll settle on PR AUC for imbalanced datasets and ROC AUC for all others.

## The Fail

### Validation

*Applied Predictive Modeling* and *Introduction to Machine Learning with Python* both make a point of stating that choosing a threshold should be done on a separate validation set - not the training set or the test set used to evaluate performance.

I'm not exactly sure what this means or how to go about it. When using cross-validation, the test set within the cross-validation splits would be the test set used to evaluate performance. But I don't really want to use my hold-out test set to set the threshold, I'd be out of test sets that the model hadn't seen and thus wouldn't have a way to verify the results.

### Pulling data

Figuring out how to get the data in openml.org was surprisingly difficult. All the datasets are saved as arff files, an ASCII text file type developed for use with Weka. My tool of choice, `pandas`, doesn't have a native way to load arrf files. Enter the `openml` package and `fetch_openml` module within `sklearn.datasets` package.

I had enough trouble figuring out the two of these packages that I address how to use them in the next entry.

### Suspiciously high precision and recall

The breast cancer dataset (both the version in Scikit-Learn and on openml.org) had suspiciously high precision, recall, and all other metrics. Sabber suggested one of the attributes probably has severe data leakage. Since this isn't what I'm exploring, I just loaded several other datasets and applied the code I had worked out using the breast cancer data. The three other datasets returned values much more in line with expectations for model performance.

### Threshold method

*Hands-On Machine Learning* used `decision_function` for the `method` parameter in `cross_val_predict` to demonstrate precision and recall using the SGDClassifier from the `sklearn.linear_model` module. The chart in the book shows thresholds ranging between around -600,000 and 600,000. *Introduction of Machine Learning* also appears to use `decision_function` to demonstrate the precision recall curve for the SVC model it trained.

Only after trying to insert a marker for the default threshold (probability of 0.5) did I really internalize that the thresholds didn't fall between 0 and 1, which is what I was expecting.

The `method` parameter of `cross_val_predict` generally accepts one of three values (it depends on the loss functions of the underlying algorithm):
- `predict`
- `predict_proba`
- `decision_function`

`predict_proba` is the one I was expecting it to return. This is the model's probability prediction that the observation belongs to the positive class. By default, values over 0.5 are grouped in the positive class and values under 0.5 and grouped in the negative class. The higher or lower the number the more confident the model is of the prediction (0.9 is a high confidence that the observation belongs in the positive class and 0.1 is a high confidence that the observation belongs in the negative class).

After looking at the examples in the books, I still had no idea what `decision_function` was. Internet to the rescue. A quick Google search turned up a [stackoverflow page](https://stackoverflow.com/questions/36543137/whats-the-difference-between-predict-proba-and-decision-function-in-scikit-lear) and a [stats.stackexchange page](https://stats.stackexchange.com/questions/329857/what-is-the-difference-between-decision-function-predict-proba-and-predict-fun). `decision_function` has to do with the hyperplane for support vector machines (SVMs). SVMs find the line/hyperplane that maximizes the distance between positive and negative classes. The threshold returned by `decision_function` is the distance of the point from the hyperplane. The examples I'd been looking at in the book were both SVMs.

The stackoverflow answer pointed out that for SVMs `predict_proba` is computed using Platt scaling which is expensive for large datasets and [is known to have theoretical issues](https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities). As such, when using SVMs the `method` parameter should be set to `decision_function`.

The `predict` parameter just returns a 1 or 0 where the threshold is the default 0.5.

### Convergence error

So, based on the number of headers under The Fail, it's obvious that this entry took more than the targeted 1-2 hours. But this is the last Fail, I swear.

While running the last dataset (much larger than the others that I'd run) I got a convergence error. Once I got past the "I'm so close!" frustration, it was an easy fix. I just upped the number of `max_iter` by a factor of 10 (default is 1,000, I specified 10,000). Problem solved. It didn't even take forever to run the cross-validation.

### Resources

- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)
- [Github page for Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow ](https://github.com/ageron/handson-ml2)
- [Area under Precision-Recall Curve (AUC of PR-curve) and Average Precision (AP)](https://stats.stackexchange.com/questions/157012/area-under-precision-recall-curve-auc-of-pr-curve-and-average-precision-ap)
- [sklearn.metrics.auc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html)
- [Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Predicting Good Probabilities With Supervised Learning](http://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- [What is the difference between decision_function, predict_proba, and predict function for logistic regression problem?](https://stats.stackexchange.com/questions/329857/what-is-the-difference-between-decision-function-predict-proba-and-predict-fun)
- [What's the difference between predict_proba and decision_function in scikit-learn?](https://stackoverflow.com/questions/36543137/whats-the-difference-between-predict-proba-and-decision-function-in-scikit-lear)
- [1.4.1.2. Scores and probabilities](https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities)
- [ConvergenceWarning: Liblinear failed to converge, increase the number of iterations](https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati)


```python

```
