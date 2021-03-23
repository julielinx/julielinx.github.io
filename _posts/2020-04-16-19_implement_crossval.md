---
title: "Entry 19: Implementing Cross-validation"
categories:
  - Blog
tags:
  - model-eval
  - dataset auto mpg
  - machine learning
---

In [Entry 18](https://julielinx.github.io/blog/18_crossval/) I finalized the decision to use a hybrid approach to validating models. Now I have to implement it.

The notebook where I did my code for this entry can be found on my github page in the [Entry 19 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/19_nb_implement_crossval.ipynb).

## The Problem

I need to split out a hold-out set, then perform stratified k-fold cross-validation on the training set. Scikit-Learn has a pretty close visual representation of this [hybrid approach](https://scikit-learn.org/stable/modules/cross_validation.html):

![hybrid cross-validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

To do this, I must complete the following steps:

- Load the data
- Split out a test set
- Segment the train set into k-folds
- Pre-process the train set of each fold
- Train the model using the train set of each fold
- Return the individual scores for each split
- Return the average score for the model

## The Options

Scikit-Learn has several options for obtaining a score using cross-validation. I'll skip the methods I already ruled out (shuffle-split, repeated, etc). The remaining eligible options include:

- [cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score)
- [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate)
- [cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html#sklearn.model_selection.cross_val_predict)
- [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold)
- [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit)

### `cross_val_score`

The limitations of this function spawned the need for [Entry 16](https://julielinx.github.io/blog/16_model_eval_and_mathjax/), basically the Define the Problem entry of this series of entries.

This function creates an internal model to determine the predictions and return a score. It is not intended to create a model which would then be applied to predict the value of new observations. 

In addition to the above problem, this function only returns a single metric at a time. I'm interested in looking at all the metrics listed in [Entry 16](https://julielinx.github.io/blog/16_model_eval_and_mathjax/).

It does allow pre-processing to be incorporated into the process via the pipeline module. As discussed in The Fail section of [Entry 17](https://julielinx.github.io/blog/17_resampling/), pre-processing on the full training set prior to splitting the data introduces data leakage.

### `cross_validate`

This function addresses one of the major failings of `cross_val_score`, it will return multiple metrics. It can also be configured to return the scores from the training data as well as the testing data and can be easily incorporated into the `pipeline` function, which will be covered in [Entry 20](https://julielinx.github.io/blog/20_sklearn_pipeline/).

### `cross_val_predict`

This function returns the prediction for each observation. Because it's part of a cross-validation function, this means that each prediction is from when that particular observation was assigned to the test set (i.e., the predictions aren't all made from the same train set values). Due to this unique return, it can only be used with cross-validation methods that assign an observation to the test set exactly once (i.e. k-fold. Not repeated, bootstrap, or shuffle).

Scikit-Learn includes a warning concerning the use of this function: "`cross_val_predict` simply returns the labels (or probabilities) from several distinct models undistinguished. Thus, `cross_val_predict` is not an appropriate measure of generalisation error."

This function is only recommended for two types of situation:

- Visualization of predictions obtained from different models
- Model blending: When predictions of one supervised estimator are used to train another estimator in ensemble methods

### `KFold`

This is the base function to split a dataset into k-folds. It allows for more control of the splitting process than the three generic functions above. It can be used in conjunction with the above functions by assigning the output of KFold to a variable, then using that variable in the `cv` parameter in the `cross_val_score`, `cross_validate`, or `cross_val_predict` function.

### `StratifiedKFold`

This is the same basic function as KFold, but which stratifies the folds.

### `TimeSeriesSplit`

This function takes into account something that I haven't addressed yet, the correlation between observations that are close together in time. The stock market is a straight forward example of when time needs to be taken into account. The observation of one day depends, at least in some part, on the observation of the prior day. For the stock market, the stock price at the end of day 1 sets the start price for day 2.

This can also be seen for behavioral trends, especially the kinds of behavior that adapt to changes. For example, if a website changes their layout, customer behavior will adapt to the changes, which will necessarily change the way a model predicts behavior.

I'm going to leave  out this consideration for now, because I'd like to explore time series and its effects more closely in another series of entries.

## The Proposed Solution

For most classification problems I work on, stratifiedKFold with shuffling seems like the best data splitting method, for regression problems, KFold seems like the obvious choice. The number of splits, per the discussion in [Entry 18](https://julielinx.github.io/blog/18_crossval/), the number of splits should be 5 or 10. I'll want to play with these two values once I start running metrics to see which works better for specific kinds of datasets / data characteristics.

The `cross_validate` function looks like the most appropriate solution to return multiple metrics. In the notebook for this entry, I'll just do some standard metrics, then explore each one in depth in later entries.

As an alternative, I could use the `.split` method of `KFold` or `StratifiedKFold` to generate indices for the split. Having the indices would allow me to use `cross_val_predict` to generate predictions. Once I have the predictions, since I have the indices, I could run whatever metrics I want on the predictions from each of the splits. The necessity of this approach will depend on whether I can run all the metrics I'm interested in using the parameters in `cross_validate`.

The last piece will be to incorporate a pipeline to allow pre-processing transformations to be completed on the training set of each split. I'll address the pipeline aspect in the next entry.

## The Fail

For some reason the `cross_validate` function requires the estimator to be a variable, not the actual algorithm function. That's strange, but not hard to overcome. Just assign the value to a variable, then drop that in.

## Up Next

[Pipelines](https://julielinx.github.io/blog/20_sklearn_pipeline/)

## Resources

- [3.1. Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
