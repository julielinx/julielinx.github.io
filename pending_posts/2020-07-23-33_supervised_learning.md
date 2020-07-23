---
title: "Entry 33: Supervised Learning"
categories:
  - Blog
tags:
  - algorithms
  - supervised learning
  - dataset forge
  - dataset wave
  - dataset breast cancer
  - dataset boston housing
  - dataset california housing
  - dataset mnist
---

## Description

In supervised learning, the data that the model is trained on includes a label for each observation where the label is the correct answer. For a numerical dataset like the [Boston housing prices dataset](https://www.kaggle.com/c/boston-housing), the label would be the price of the house. For a categorical dataset such as the [iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), the label would be the species of the flower.

[Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) puts it succinctly on page 27:

> supervised learning is used whenever we want to predict a certain outcome from a given input, and we have examples of input/output pairs.

Supervised learning is generally broken into two categories based on the type of labels:

- Regression
- Classification

### Regression

For regression, the goal is to predict a continuous number. The above referenced Boston housing prices dataset is an example of a regression problem. Fitness provides a plethora of examples of regression problems. If I want to try to predict heart rate, the number of calories burned, or the age of the participant those would all be regression problems.

### Classification

For classification, the goal is to predict a category. An example of a classification problem is the above referenced iris dataset. The categories are the type of flower. Another example of classification is the [titanic dataset](https://www.kaggle.com/c/titanic). People either survived the sinking of the ship or they didn't.

An important distinction within classification is *binary classification* vs *multiclass classification*.

- Binary classification
  - There are two and only two labels.
  - Example: Titanic dataset. Survived: yes or no.
- Multiclass classification
  - There are more than two labels.
  - Example: iris dataset. Species of flower:  Setosa, Versicolour, or Virginica

*Side note*, if there is only one label, the problem isn't really a prediction problem - everything is the same.

There is another type of classification, *multilabel classification*. An example of this would be browser used by a customer for an online retailer. There's nothing stopping the customer from using more than one browser and so the observation for that customer would have more than one label. I haven't worked with multilabel problems, and they don't seem particularly prevalent, so this will probably be as in-depth as I go into this particular topic.

### Regression vs multiclass classification

A situation could arise wherein there are so many categories and the categories are numeric, that it becomes difficult to tell if the problem should be categorized as regression or multiclass. If the values of the label are continuous or fall along a scale (like centimeters on a ruler) then the problem is a regression problem. If the values are discrete and separate, the problem is a classification problem.

For example, if the label included the ages of children between 5 and 18 years, it may be tempting to think of this as a classification problem. However, because the ages fall on a scale, it should be a regression problem. On the other hand, if we had grouped the children into child, youth, preteen, and teenager then it would be a classification problem.

For more on determining numeric vs categorical, check out the Types of Variables section of [Entry 13](https://julielinx.github.io/blog/13_cat_prelims/).

## Purpose

The purpose of supervised learning is to take information that is known and generalize patterns in that information to predict values for data that has never been seen before.

## Limitations

For this learning style, every observation in the training dataset must have a label. To appropriately evaluate how the model performed, every observation in the testing dataset must also have a label. However, a label isn't required for the express purpose of making a prediction - just determining how good the prediction is.

## Datasets

### Introduction to Machine Learning with Python

The `mglearn` package has two small synthetic/artificial datasets:

- Forge
  - Binary classification
  - X, y = mglearn.datasets.make_forge()
- Wave
  - Regression
  - X, y = mglearn.datasets.make_wave(n_samples=40)

The book also recommends two larger, real-world datasets available in Scikit-Learn:

- Wisconsin Breast Cancer
  - Binary classification
  - from sklearn.datasets import load_breast_cancer
  - cancer = load_breast_cancer()
- Boston Housing
  - Regression
  - from sklearn.datasets import load_boston
  - boston = load_boston()
  - Extended dataset available in `mglearn` with over 100 features
  - X, y = mglearn.datasets.load_extended_boston()
  
### Hands-On Machine Learning

- California Housing dataset
  - Regression
  - He stores it in a location he controls, but it's easier to get programmatically from `sklearn`
  - from sklearn.datasets import fetch_california_housing
  - housing = fetch_california_housing()
- MNIST
  - Classification - multiclass
  - from sklearn.datasets import fetch_openml
  - mnist = fetch_openml('mnist_784', version=1)
  
## Up Next

Datasets

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
