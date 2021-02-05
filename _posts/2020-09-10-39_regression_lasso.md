---
title: "Entry 39: Lasso Regression"
categories:
  - Blog
tags:
  - regression
  - supervised learning
  - regularization
  - machine learning
---

Least Absolute Shrinkage and Selection Operator (LASSO) Regression is a form of regression regularization using L1 regularization.

The notebook where I did my code for this entry can be found on my github page in the [Entry 39 notebook](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/01_regression/39a_lasso_regression.ipynb).

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

This method similar to Ridge Regression <font color='red'>Entry 40</font> in that it uses the same formula as OLS and minimizes the magnitude of coefficients (*w*), but instead of using L2 regularization, Lasso Regression uses L1 regularization.

## Purpose

On pages 137 and 140, [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) provides the following two updated equations from the OLS linear regression versions:

- Cost function: $J(\theta) = MSE(\theta) + \alpha \displaystyle\sum_{i=1}^n \lvert \theta_{i}\rvert$
  - Note: The bias term $\theta_{0}$ is not regularized as indicated by the start of $i$ at 1 instead of 0
  - Where
    - MSE: mean squared error
    - $\alpha$: regularization term
      - i.e. how much the model is regularized
      - A value of 0 is no regularization (i.e. regular linear regression)
      - As the value approaches infinity, the weights approach zero (i.e. no features contribute and results in a flat line through the data's mean)
    - $\theta$: the theta array
- Lasso regression subgradient vector: $g(\theta, J) = \nabla_{0} \mathrm{MSE}(\theta) + \alpha 
  \begin{pmatrix}
    \mathrm{sign}(\theta_{1}) \\
    \mathrm{sign}(\theta_{2}) \\
    \vdots \\
    \mathrm{sign}(\theta_{n})
  \end{pmatrix}$
  - Where
    - $g(\theta, J)$: subgradient vector
    - $ \nabla_{0}$: differential operator (the symbol is called "nabla")
    - $\mathrm{MSE}$: mean squared error
    - $\alpha$: the regularization term
      - i.e. how much the model is regularized
      - A value of 0 is no regularization (i.e. regular linear regression)
      - As the value approaches infinity, the weights approach zero (i.e. no features contribute and results in a flat line through the data's mean)
    - $\mathrm{sign}(\theta_{i}) = 
  \begin{cases}
    -1   & \mathrm{if } \; \theta_{i} < 0 \\
    \;\,\,0    & \mathrm{if } \; \theta_{i} = 0 \\
    +1   & \mathrm{if } \; \theta_{i} > 0
  \end{cases}$

## Behavior

The outcome of keeping the coefficients as small as possible helps reduce the complexity of the model (i.e. makes it more generalized and reduces overfitting). The complexity of the model is controlled through the parameter `alpha`. Larger `alpha` values force the coefficients closer to 0 (reducing complexity), smaller values are less restricting (allowing for more complexity).

[Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) puts it succinctly on page 55:

> The consequence of L1 regularization is that when using the lasso, some coefficients are *exactly zero*. This means some features are entirely ignored by the model. This can be seen as a form of automatic feature selection.

## Parameters

- `alpha`
- `max_iter`

## Strengths

Having some of the coefficients be exactly zero can make a model easier to interpret and make more important features stand out. Coefficients that are 0 can also function as an automated form of feature reduction.

Minimizes variance.

## Limitations

- If `alpha` is set too high, very few features will remain and the model will underfit the data
- May behave erratically when the number of features is greater than the number of training instances
- May behave erratically when several features are strongly correlated

## Up Next

[Ridge Regression](https://julielinx.github.io/blog/40_regression_ridge/)

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
