---
title: "Entry 38: Regularization"
categories:
  - Blog
tags:
  - regression
  - regularization
  - overfitting
  - supervised learning
---

Regularization is used to help address overfitting.

## Description

There are basically two strategies:

- Reduce the magnitude/values of the theta array
  - This method retains all features
  - It works well when there are a lot of features and each contirbutes at least marginally to the prediction ability
- Reduce the number of features
  - This method removes features
  - It can be done two ways:
    - Manually: select which features to keep by hand
    - Automatatically: use mathematics to automate feature selection
    
### L1 and L2 Regularization

L1 and L2 regularization are common regularization techinques. Each of the techniques covers one of the regularzation strategies. They are based on L1 and L2 norms. The thing that helped me understand why they're called L1 and L2 was in a [Towards Data Science article](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261):

- $\text{L1 norm} = \lvert \lvert w \rvert \rvert_{1} = \lvert w_{1} \rvert + \lvert w_{2}\rvert + \dotsb + \lvert w_{n}\rvert$
- $\text{L2 norm} = \lvert \lvert w\rvert \rvert_{2} = \sqrt{\lvert w_{1}\rvert ^2 + \lvert w_{2}\rvert ^2 + \dotsb + \lvert w_{n}\rvert^2}$
- $\text{Lp norm} = \lvert \lvert w\rvert \rvert_{p} = \sqrt[p]{\lvert w_{1}\rvert ^p + \lvert w_{2}\rvert ^p + \dotsb + \lvert w_{n} \rvert ^p}$

They are basically all the same equation (the third one), but to different powers: 1, 2, and n.

## Purpose

A regularization term is added to the cost function, which makes it look like the feature is more incorrect than it actually is, which lowers the theta term giving the feature less weight.

Cost functions from [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) on pages 114, 137, and 135:

- Base cost function:
  - $J(\theta) = MSE(\theta) = \frac{1}{m} \displaystyle\sum_{i=1}^m (\theta^{T}x^{(i)} - y^{(i)})^{2}$
- Cost function with L1 regularization:
  - $J(\theta) = MSE(\theta) + \alpha \displaystyle\sum_{i=1}^n \lvert \theta_{i}\rvert$
- Cost function with L2 regularization:
  - $J(\theta) = MSE(\theta) + \alpha \frac{1}{2} \displaystyle\sum_{i=1}^n \theta_{i}^{2}$

## Behavior

At first I was like "Why are you *adding* the penalty? Won't that make the weight larger? Shouldn't you be *subtracting* the penalty and making the weight smaller?"

Here's how I think about it:

- The weights (i.e., the theta array) show which features contribute most strongly to the prediction; The larger the weight, the more important that feature.
  - This concept is exemplified in [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0) on page 101, during the discussion on intrepretability:
      > [...] if the estimated coefficient of a predictor is 2.5, then a 1 unit increase in that predictor's value would, on average, increate the response by 2.5 units.
      - Where the estimate coefficient is the same as what I've been referring to as a weight in the theta array.
- Features are more important (i.e., higher weights) when the predicted value is closer to the observed value ($\frac{1}{n} \sum (y_{i} - \hat{y_{i}})^2$ (i.e. mean squared error) is small.
- Features that are less important (i.e., lower weights) when the predicted value is farther from the observed value ($\frac{1}{n} \sum (y_{i} - \hat{y_{i}})^2$ (i.e. mean squared error)  is large).

#### "These aren't the droids you're looking for."

The key here is that the penalty is added to the cost function, or mean squared error, not the weight. This makes the feature look like it's worse than it is and is thus is assigned a lower weight.

## Strengths and Weaknesses

Information from Medium article [L1 and L2 Regularization](https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2):

<table align='left'>
    <tr>
        <td><b>L1 Regularization</b></td>
        <td><b>L2 Regularization</b></td>
    </tr>
    <tr>
        <td>Penalizes sum of absolute value of weights</td>
        <td>Penalizes sum of square weights</td>
    </tr>
    <tr>
        <td>Sparse solution</td>
        <td>Non sparse solution</td>
    </tr>
    <tr>
        <td>Has multiple solutions</td>
        <td>Has one solution</td>
    </tr>
    <tr>
        <td>Automatic feature selection</td>
        <td>Retains all features</td>
    </tr>
    <tr>
        <td>Robust to outliers</td>
        <td>Not robust to outliers</td>
    </tr>
    <tr>
        <td>Generates models that are simple and interpretable</td>
        <td>Gives better predictions when output variable is a function of all input features</td>
    </tr>
    <tr>
        <td>Can't learn complex patterns</td>
        <td>Can learn complex data patterns</td>
    </tr>
</table>

## Up Next

[Lasso Regression](https://julielinx.github.io/blog/39_regression_lasso/)

## Resources

- [Intuitions on L1 and L2 Regularisation](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [L1 and L2 Regularization](https://medium.com/datadriveninvestor/l1-l2-regularization-7f1b4fe948f2)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
