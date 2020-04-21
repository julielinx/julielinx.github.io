---
title: "Entry 25: Baseline Models"
categories:
  - Blog
tags:
  - model-eval
---

As discussed in <font color='red'>Entry 16</font>, certain characteristics in the data can make the model look like it's performing better than it is. One of these characteristics is class imbalance, wherein one class is much more prevalent in the data. By always predicting the majority class, the model can seem like it's performing very well.

A simple sanity check is to compare the model performance against a naive dummy model.

## The Problem

In order to determine how well or poorly a model is performing, some kind of baseline must be established. This can represent random guessing, an educated guess like the average (regression) or most common (logistic) value, a constant value, etc.

## The Options

The recommendation of [Chris Albon](https://chrisalbon.com/) in *[Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)* is to use a baseline dummy model, which Scikit-learn's Dummy estiamtors do for me.

## The Proposed Solution

I used the DummyRegressor and the DummyClassifier to create baseline models. Each of the options has multiple choices for the naive solution to use (constant, median, most frequent, etc). By creating a little function I was able to quickly and easily try all of the options that didn't require parameters.

## The Fail

I incorporated `cross_validate` with multiple scoring parameters into the function to assess the dummy model but it was too messy and cluttered. A single score worked just fine for this initial test.

I thought about building out a full classification model, where I encode the categoricals and all that. However, I'm going to leave this to address in the final post of the series. Then I can practice incorporating all the different steps when I start working my way through the different algorithms.

This last point isn't so much a fail as a confession. I did the dummy models in the <font color='red'>Entry 24 notebook</font>. However, I wanted to be able to easily find it for future reference, so I broke it out into it's own post, but the code I figured out while writing the previous notebook.

## Up Next

Thresholds - PR and ROC

### Resources

- [Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)
- [3.3.6. Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
- [sklearn.dummy.DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html?highlight=dummyclassifier#sklearn.dummy.DummyClassifier)


```python

```
