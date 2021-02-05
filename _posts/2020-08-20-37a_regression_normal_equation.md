---
title: "Entry 37a: Normal Equation"
categories:
  - Blog
tags:
  - regression
  - regression calculation
  - supervised learning
  - machine learning
---

The Normal Equation generates the weights that are used in Linear Regression. It calculates the theta array directly without having to iterate through different solutions (the way Gradient Descent does).

## Purpose

The feature matrix, its transpose, and its inverse are used in conjunction with the target array to directly calculate the weights of the features.

$\hat{\theta} = (X^{T} X)^{-1} X^{T} y$

Where:

- $\hat{\theta}$: theta array, the hypothesized weights
- X: input feature matrix
- $X^{T}$: the transpose of X
- y: array of target values

## Behavior

An interesting paragraph in [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) sent me into the Scikit-learn documentation to actually read the full page for `sklearn.linear_model.LinearRegression`. The documentation states:

> From the implementation point of view, this is just plain Ordinary Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.

As a reminder from [Entry 36](https://julielinx.github.io/blog/36_regression_OLS/), using the Normal Equation to calculate the theta array requires the inverse of the matrix, and there isn't always a solution for the inverse.

$\hat{\theta} = (X^{T} X)^{-1} X^{T} y$

When there isn't a solution for the inverse, Gradient Descent has to be used instead.

However, because of the way the Normal Equation is implemented in `scipy.linalg.lstsq` this limitation is no longer a problem. The reason is a little mathy, but it comes down to the fact that the function uses the *pseudoinverse* instead of the inverse.

$\hat{\theta} = X^{+} y$

Here's what *Hands-On Machine Learning with Scikit-Learn* has to say about it:

> The pseudoinverse itself is computed using a standard matrix factorization technique called *Singular Value Decomposition* (SVD) that can decompose the training set matrix **X** into the matrix multiplication of three matricies **U** **$\sum$** **$V^{t}$** (see `numpy.linalg.svd()`). The psudoinverse is computed as $X^{+} = V \sum^{+} U^{T}$. To compute the matrix $\sum^{+}$, the algorithm takes $\sum$ and sets to zero all values smaller than a tiny threshold value, then it replaces all the nonzero values with their inverse, and finally it transposes the resulting matrix. This approach is more efficient that computing the Normal Equation, plus it handles edge cases nicely; indeed, the Normal Equation may not work if the matrix $X^{t}X$ is not invertible (i.e. singular), such as if *m < n* or if some features are redundant, but the pseudoinverse is always defined.

The important takeaways I get from that are:

- SVD is used to calculate the pseudoinverse
  1. All values smaller than a threshold are set to 0
  2. All non-zero values are placed with their inverse
  3. The matrix is transposed
- SVD is more efficient than the straight Normal Equation
- The pseudoinverse can always be defined (unlike the inverse, which may not have a solution)

## Up Next

[Gradient Descent](https://julielinx.github.io/blog/37b_regression_gradient_descent/)

## Resources

Andrew Ng does a fantastic job of covering the theory and practicalities of gradient descent in weeks 1 and 2 of his [Machine Learning course](https://www.coursera.org/learn/machine-learning) on coursera. For more details, watch the videos, read the notes, and (most telling) take the quizes for those two weeks.

- [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Machine Learning course](https://www.coursera.org/learn/machine-learning)