---
title: "Entry 41: Elastic Net"
categories:
  - Blog
tags:
  - regression
  - supervised learning
---

The notebook where I did my code for this entry can be found on my github page in the [Entry 41 notebook](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/41a_elasticnet_regression.ipynb).

## Learning Style

<table align='left'>
    <tr>
        <th>Supervision</th>
        <th>Prediction types</th>
    </tr>
    <tr>
        <td>Supervised</td>
        <td>Regression</td>
    </tr>
</table>

## Description

This is another form of regression regularization. It combines the penalties of Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization).

[Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) states on page 57 into 58 that this combination of L1 and L2 usually works best in practice.

## Purpose

On page 140, [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) provides the following equation for the cost function of Elastic Net:

$J(\theta) = MSE(\theta) + r \alpha \displaystyle\sum_{i=1}^n \lvert \theta_{i}\rvert + \frac{1 - r}{2} \alpha \displaystyle\sum_{i=1}^n \theta_{i}^{2}$

Where

- $J(\theta)$: cost function
- MSE: mean squared error
- $r$: mix ratio
  - The range of values is 0 to 1
  - *Note*, at the extremes (0 or 1) one or the other of the terms cancels out (i.e. it's 0)
  - When r = 0, Elastic Net is the same as Ridge Regression
    - $r \alpha \displaystyle\sum_{i=1}^n \lvert \theta_{i}\rvert$ cancels to 0
  - When r = 1, Elastic Net is the same as Lasso Regression
    - $\frac{1 - r}{2} \alpha \displaystyle\sum_{i=1}^n \theta_{i}^{2}$ cancels to 0
- $\alpha$: regularization term
  - i.e. how much the model is regularized
  - A value of 0 is no regularization (i.e. regular linear regression)
  - As the value approaches infinity, the weights approach zero (i.e. no features contribute and results in a flat line through the data's mean)
- $\theta$: the theta array

## Behavior

In `scikit-learn`, the ratio of how the Lasso (L1) and Ridge (L2) terms are applied is controlled by the `l1_ratio` parameter. In other words, the `l1_ratio` parameter specifies how much Lasso is used. A value of 0 would mean the Elastic Net would be the same as Ridge Regression, a value of 0.5 would be half Lasso and half Ridge, and a value of 1 would be the same as Lasso Regression.

The outcome of keeping the coefficients as small as possible helps reduce the complexity of the model (i.e. makes it more generalized and reduces overfitting). The complexity of the model is controlled through the parameter `alpha`. Larger `alpha` values force the coefficients closer to 0 (reducing complexity), smaller values are less restricting (allowing for more complexity).

## Parameters

- `alpha`
- `l1_ratio`
- `max_iter`

## Strengths

- Doesn't suffer the same problems as Lasso in regards to erratic behavior when the number of features is greater than the number of training instances or when several features are strongly correlated
- Doesn't retain all features like Ridge Regression, which helps automatically eliminate useless features

## Limitations

- At the extreme values of $r$, Elastic Net can suffer the same problems as either Lasso Regression or Ridge Regression
- There are multiple parameters to tune

## Up Next

[Logistic Regression](https://julielinx.github.io/blog/42_regression_logistic/)

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
