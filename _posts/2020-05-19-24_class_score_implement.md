---
title: "Entry 24: Scoring Classification Models - Implementation"
categories:
  - Blog
tags:
  - model-eval
  - classification models
  - dataset breast cancer
---

Now that I've got a handle on the measurement options and equations for classification problems, it's time to implement those measures on actual models.

The notebook where I did my code for this entry can be found on my github page in the [Entry 24 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/24_nb_class_score_implement.ipynb).

## The Problem

In [Entry 23](https://julielinx.github.io/blog/23_class_score_theory/) I covered the mathematical options for measuring classification models. But just like entries [21](https://julielinx.github.io/blog/21_reg_score_theory/) and [22](https://julielinx.github.io/blog/22_reg_score_implement/), just because I know the equations doesn't mean I can apply it to actual data.

## The Options

### `scoring` parameter

The first option is to list scoring methods in the scoring parameter of `cross_validate` like I did for the regression metrics in the [Entry 22 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/22a_nb_reg_score_implement.ipynb). As a reminder, the list of available parameters is in the [3.3.1. The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) section of the Scikit-Learn documentation.

The same restrictions that applied to the regression metrics also apply to the classification metrics, they're limited to those that don't require extra parameters. The available classification metrics are:

- `accuracy`
- `balanced_accuracy`
- `roc_auc`
- `roc_auc_ovr`
- `roc_auc_ovo`
- `roc_auc_ovr_weighted`
- `roc_auc_ovo_weighted`
- `neg_log_loss`
- `neg_brier_score`
- `precision`
- `average_precision`
- `precision_macro`
- `precision_micro`
- `precision_samples`
- `precision_weighted`
- `recall`
- `recall_macro`
- `recall_micro`
- `recall_samples`
- `recall_weighted`
- `f1`
- `f1_macro`
- `f1_micro`
- `f1_samples`
- `f1_weighted`
- `jaccard`
- `jaccard_macro`
- `jaccard_micro`
- `jaccard_samples`
- `jaccard_weighted`

From the list it's easy to see there are five versions of some of these metrics. Fortunately, they follow a standard naming convention which is spelled out in the [3.3.2.1. From binary to multiclass and multilabel](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel) section of the Scikit-Learn documentation. Based on the definitions (and the section title), the variations are used on multiclass or multilabel problems.

> - `macro` simply calculates the mean of the binary metrics, giving equal weight to each class. In problems where infrequent classes are nonetheless important, macro-averaging may be a means of highlighting their performance. On the other hand, the assumption that all classes are equally important is often untrue, such that macro-averaging will over-emphasize the typically low performance on an infrequent class.
> - `weighted` accounts for class imbalance by computing the average of binary metrics in which each class’s score is weighted by its presence in the true data sample.
> - `micro` gives each sample-class pair an equal contribution to the overall metric (except as a result of sample-weight). Rather than summing the metric per class, this sums the dividends and divisors that make up the per-class metrics to calculate an overall quotient. Micro-averaging may be preferred in multilabel settings, including multiclass classification where a majority class is to be ignored.
> - `samples` applies only to multilabel problems. It does not calculate a per-class measure, instead calculating the metric over the true and predicted classes for each sample in the evaluation data, and returning their (sample_weight-weighted) average.

There are two metrics in the above list that weren't in my review of my machine learning books: log loss and brier score.

#### Log loss

This metric returns the negative log-likelihood of the classifier given the true label.

$L_{log}(y,p) = -\text{logPr}(y \lvert p) = -(y\text{log}(p) + (1-y)(\text{log}(1-p))$

It's easiest to see in the last equation that one side or the other of the equation cancels out.

There is a good explanation of log loss on the [Wiki fast](http://wiki.fast.ai/index.php/Main_Page) entry for [Log Loss](http://wiki.fast.ai/index.php/Log_Loss).

The examples on that page are:

For a given class label of 1 and a predicted probability of .25:

- $-{(1\log(.25) + (1 - 1)\log(1 - .25))}$
- $-{(\log(.25) + 0\log(.75))}$
- $-{\log(.25)}$

For a given class label of 0 and a predicted probability of .25:

- $-{(0\log(.25) + (1 - 0)\log(0 - .25))}$
- $-{(1\log(-.25))}$
- $-{\log(-.25)}$

The metric is designed to penalize both type I and type II errors, but more so to discriminate against predictions that are confident about their wrong prediction. Wiki fast provides the following visualization of this concept:

![log loss penality chart](http://wiki.fast.ai/images/4/43/Log_loss_graph.png)

In the chart, the actual value is 1. When the probability is very low (the left side of the chart), the log loss value is high. As the probability rises, the log loss quickly decreases to more moderate values and is slow to approach 0 (predicted perfectly).

#### Brier score

This is the difference between the probability assigned to the prediction and the actual outcome.

*brier score* $= \frac{1}{n} \sum{(f_{t} - o_{t})}^2$

Where:
- *n* = the total number of predictions
- $f_{t}$ = the predicted probability
- $o_{t}$ = the actual outcome

Basically, for each prediction a probability is assigned. The class the sample is assigned to depends on the threshold for the probability. For simplicity sake, I'll say that anything above 0.5 is assigned to 1 (the true class) and everything below 0.5 is assigned to 0 (the negative class - don't get technical on me about values that are exactly 0.5, this is a *simple* example).

So if the probability was 0.65 the sample is assigned to 1. Let's say this is correct and there are 50 total samples. The $\sum{(f_{t} - o_{t})}^2$ portion of the equation would look as follows:

$(0.65 - 1)^2$

The same calculation would be applied to each of the 50 samples, then all added together, and finally divided by *n*.

The values range from 0 to 1. As the [brier_score_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) documentation puts it "the lower the Brier score is for a set of predictions, the better the predictions are calibrated." 

### `metrics` module

The second option is the list of classification metrics available in the `metrics` module. This list is more extensive than what's available by default for the `scoring` parameter. The list of functions is:

- `accuracy_score`
- `balanced_accuracy_score`
- `precision_score`
- `average_precision_score`
- `precision_recall_curve`
- `precision_recall_fscore_support`- `recall_score`
- `roc_auc_score`
- `roc_curve`
- `cohen_kappa_score`
- `confusion_matrix`
- `hinge_loss`
- `matthews_corrcoef`
- `classification_report`
- `f1_score`
- `fbeta_score`
- `hamming_loss`
- `jaccard_score`
- `log_loss`
- `multilabel_confusion_matrix`
- `zero_one_loss`

There are a few metrics on the list that I haven't defined yet.

#### Hinge loss

Based on the Scikit-Learn documentation [3.3.2.10. Hinge loss](https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss) and [sklearn.metrics.hinge_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hinge_loss.html#sklearn.metrics.hinge_loss) as well as this [medium.com article](https://medium.com/analytics-vidhya/understanding-loss-functions-hinge-loss-a0ff112b40a1), this [towardsdatascience.com article](https://towardsdatascience.com/support-vector-machines-intuitive-understanding-part-1-3fb049df4ba1), and [this blog](https://jamesmccaffrey.wordpress.com/2018/10/04/hinge-loss-explained-with-a-table-instead-of-a-graph/) hinge loss is generally used with Support Vector Machines (SVMs). The examples are all for values of +1 and -1. The purpose of hinge loss is for "maximum-margin" classification (to place the plane of separation where there is the most space) and considers only prediction errors.

$L_{Hinge}(y, w) = max(1 - wy, 0) = \lvert 1 - wy \rvert_{+}$

Where:
- *y* = true value
- *w* = predicted probability

This metric seems specialized to work with SVMs and to have the same general purpose as log loss.

#### fbeta score

This is the same basic metric as the $F_{1}$ score, but adds a parameter, `beta`, that allows for weighting precision or recall.

A parameter of `beta` < 1 gives more weight to precision and `beta` > 1 gives more weight to recall. At the extremes `beta` = 0 only considers precision and `beta` = +inf only considers recall.

$F_{\beta} = (1+\beta^2) \frac{precision \times recall}{\beta^2 precision + recall}$

#### Hamming loss

Based on the [Wikipedia entry](https://en.wikipedia.org/wiki/Hamming_distance), hamming loss is generally used to determine the distance between two strings. Or as the entry rephrases:

> "In other words, it measures the minimum number of substitutions required to change one string into the other, or the minimum number of errors that could have transformed one string into the other."

$L_{hamming}(y, \hat{y}) = \frac{1}{n_{labels}} \sum{1(\hat{y_{i}} \neq y_{i})}$

Where
- $n_{labels}$ is the number of classes (or labels)

Based on the equation, it looks like this translates to loss functions by taking the average number of mistakes per class. $\sum{1(\hat{y_{i}} \neq y_{i})}$ is basically a count of when the observation and prediction don't match. That is then divided by the number of classes, giving an average number of mistakes per class.

## The Proposed Solution

### `scoring` parameter

Since I'm focused on binary problems, I can ignore the micro/macro/etc variations. As discussed earlier I'm not interested in the accuracy scores due to the imbalanced classes I work with, so those are out too. I'll be addressing ROC/AUC and thresholds in [Entry 26](https://julielinx.github.io/blog/26_thresholds_pr_roc/), so they'll be dealt with there.

This leaves me with the following options:

- `neg_log_loss`
- `neg_brier_score`
- `precision`
- `average_precision`
- `recall`
- `f1`

Considering the overall number of classification metrics I'm interested in, I find it a little ironic that I'm left with fewer classification metrics than I was for regression metrics. This really just means I get to figure out how to use the `make_scorer` function.

### `metrics` module

Using the same criteria as for the `scoring` parameter option, and removing what was already covered there, my list of functions is greatly reduced. I also removed some functions like `confusion_matrix` that don't return a single scoring value. These cuts leave me with:

- `balanced_accuracy_score` (with `adjusted=True`, balanced accuracy is the Youden's J statistic/informedness)
- `cohen_kappa_score`
- `matthews_corrcoef`
- `fbeta_score`

### Overview

The metrics I decided I'm interested in while completing [Entry 23](https://julielinx.github.io/blog/23_class_score_theory/), along with where they're available are recapped below:

- No information rate: easily obtained using the dummy classifiers
- Cohen’s Kappa: `metrics` module
- Precision: `scoring` parameter
- Markedness: not available
- Recall: `scoring` parameter
- Informedness/Youden’s J index/balanced_accuracy: `metrics` module (*note* I can't use the `balanced_accuracy` option in the `scoring` parameter because I need to set the `adjusted` parameter to `True`
- Specificity: may be available via the `imbalanced-learn` package
- F1-score: `scoring` parameter
- Matthews correlation coefficient: `metrics` module
- Critical success index: not available

Specificity isn't natively available via either the `scoring` parameter or the `metrics` module. However, the internet pointed out that specificity is recall of the negative class, so it is possible to fanagle it in. There is also a package `imbalanced-learn` that has a `specificity_score` function that I may just be able to wrap in`make_scorer`.

Neither markdedness nor critical success index turned up any results with Scikit-Learn. The metrics would have to be custom created and wrapped in the `make_scorer` function. This isn't my focus, so I'll be skipping these two for now.

There were a few additional metrics that I may be interested in that were listed as options in Scikit-Learn:

- `neg_log_loss`
- `neg_brier_score`
- `fbeta_score`
- `log_loss`

## The Fail

The most obvious fail was my inability to find a way to run three of the metrics I'm interested in (markedness, specificity, critical success index). However, the goal of this series of posts is to evaluate modes, not write code from scratch. I have to draw a line somewhere to keep myself from detouring down rabbit holes.

The other fail was that after a brief foray into the `imbalneced-learn` documentation, I branded it another rabbit hole at this stage. While it may be nice to have the options available in that package, I feel that once the Precision/Recall (PR) and Receiver Operator Characteristic (ROC) curves have been incorporated into the evaluation, I'll be   able to establish a well rounded picture of model performance.

## Up Next

[Naive baseline models](https://julielinx.github.io/blog/25_baseline_compare/))

### Resources

- [Log Loss](http://wiki.fast.ai/index.php/Log_Loss)
- [Understanding binary cross-entropy / log loss: a visual explanation](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
- [Brier score](https://en.wikipedia.org/wiki/Brier_score)
- [sklearn.metrics.brier_score_loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss)
- [Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss)
- [Understanding loss functions : Hinge loss](https://medium.com/analytics-vidhya/understanding-loss-functions-hinge-loss-a0ff112b40a1)
- [Support vector machines ( intuitive understanding ) — Part#1](https://towardsdatascience.com/support-vector-machines-intuitive-understanding-part-1-3fb049df4ba1)
- [Hinge Loss Explained with a Table Instead of a Graph](https://jamesmccaffrey.wordpress.com/2018/10/04/hinge-loss-explained-with-a-table-instead-of-a-graph/)
- [How do you minimize “hinge-loss”?](https://math.stackexchange.com/questions/782586/how-do-you-minimize-hinge-loss)
- [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)
- [Entry 23: Scoring Classification Models - Theory](https://julielinx.github.io/blog/23_class_score_theory/)
- [Entry 26: Setting thresholds - precision, recall, and ROC](https://julielinx.github.io/blog/26_thresholds_pr_roc/)
