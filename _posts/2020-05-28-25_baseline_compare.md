---
title: "Entry 25: Baseline Models"
categories:
  - Blog
tags:
  - model-eval
  - dataset auto mpg
  - dataset breast cancer
  - machine learning
---

As discussed in [Entry 16](https://julielinx.github.io/blog/16_model_eval_and_mathjax/), certain characteristics in the data can make a model look like it's performing better than it is. One of these characteristics is class imbalance, wherein one class is much more prevalent. By always predicting the majority class, the model can seem like it's performing very well.

A simple sanity check is to compare model performance against a naive dummy model.

The notebook where I did my code for this entry can be found on my github page in the [Entry 25 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/25_nb_baseline_compare.ipynb).

## The Problem

In order to determine how well or poorly a model is performing, some kind of baseline must be established. This baseline could be established through random guessing, an educated guess (like the mean  for regression problems or the most common value for classification problems), a constant value, etc.

## The Options

The recommendation of [Chris Albon](https://chrisalbon.com/) in *[Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)* is to use a baseline dummy model. This is easily accomplished with Scikit-learn's Dummy estimators.

## The Proposed Solution

I used `DummyRegressor` and `DummyClassifier` to create baseline models. Each of the options has multiple choices for the naive solution to use (constant, median, most frequent, etc). By creating a little function I was able to quickly and easily try all of the options that didn't require parameters.

## The Fail

I incorporated `cross_validate` with multiple scoring parameters into the function to assess the dummy model, but it was too messy and cluttered. A single score worked just fine for this initial test.

I thought about building out a full classification model, where I handle all pre-processing like encoding categorical variables and all that. However, I'm going to leave this to address in the final post of the series. Then I can practice incorporating all the different steps when I start working my way through different algorithms.

This last point isn't so much a fail as a confession. I did the dummy models in the [Entry 24 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/24_nb_class_score_implement.ipynb). However, I wanted to be able to easily find it for future reference, so I broke it out into its own post. But I did figure out the code while writing the previous notebook.

## Up Next

Thresholds - PR and ROC

### Resources

- [Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)
- [3.3.6. Dummy estimators](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
- [sklearn.dummy.DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html)
- [sklearn.dummy.DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html?highlight=dummyclassifier#sklearn.dummy.DummyClassifier)
- [Entry 16: Model Evaluation](https://julielinx.github.io/blog/16_model_eval_and_mathjax/)