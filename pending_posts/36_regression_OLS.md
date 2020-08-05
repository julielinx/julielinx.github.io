# Entry 36 - Ordinary Least Squares (OLS)

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

On page 49, [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) calls this "the simplest and most classic linear method for regression." It is usually the default method of Linear Regression and is the method used in the `sklearn.linear_model.LinearRegression` function.

This method uses mean squared error (MSE) to find the best fit line.

I covered mean squared error in [Entry 21](https://julielinx.github.io/blog/21_reg_score_theory/), but here's a reminder along with how the equation would be applied over a dataset/matrix:

- **error**: the difference between the predicted value and the true value
- **squared error**: literally square the error term. This makes all values positive. Squaring is used instead of absolute value in order to make outlier terms more important
- **mean squared error**: sum the squared error of all data points and divide by the number of data points

$MSE(X, h_{\theta}) = \frac{1}{m} \sum (\theta^{T}x^{(i)} - y^{(i)})^2$

Where:

- $X$: matrix of features
- $h_{\theta}$: prediction function, also called a *hypothesis*; $h_{\theta} = \theta^{T}x^{(i)}$
- $\theta$: array of weights
- $x^{(i)}$: array of features for a specific observation
- $y^{(i)}$: observed output for a specific observation

## Purpose

OLS is basically the starting point for linear regression. It calculates the theta array (I.E., the list of weights) used to calculate the output (I.E., the predicted value) from an array of inputs.

There are two options when calculating the theta array:

- Normal equation
- Gradient descent

When the matrix has an inverse, it is calculated using the Normal Equation. When there is no inverse or the dataset is too large, then the iterative process of Gradient Descent is used.

### Regularization

When I think of "regularization," I generally thinking of [centering, scaling, and normalization](https://julielinx.github.io/blog/08_center_scale_and_latex/). However, in this case, *Introduction to Machine Learning using Python* defines it this way on page 51:

> **Regularization**: explicitly restricting a model to avoid overfitting.

In his [Machine Learning course](https://www.coursera.org/learn/machine-learning) Andrew Ng notes these two benefits to regularization:

- Reduces the magnitude/values of the theta array, allowing for the retention of all features (as opposed to manually or automatically removing low contributing features)
- Works well when there are a lot of features and each contirbutes at least minimally to the prediction ability

## Behavior

### Normal Equation

The Normal Equation directly calculates the theta array.

The vectorized equation is:

$\hat{\theta} = (X^{T} X)^{-1} X^{T} y$

*Side note*, a vectorized form of an equation just means that the full matrix can be used instead of inputing each array of features for every observation.

Where:

- $\hat{\theta}$: theta array, the hypothesized weights
- $X$: input feature matrix
- $X^{T}$: the transpose of X
- $y$: array of target values

I go into more detail on the Normal Equation in the <font color='red'>next entry</font>.

### Gradient Descent

Gradient Descent starts with an initial theta array, then iterates through improving the fit of the theta array with each iteration.

There are several different ways to write out the gradient descent equation. The main idea of the equation as written in the *Machine Learning course* is:

$\theta_{j} := \theta_{j} - \alpha \frac{1}{m}\displaystyle\sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}$

Where:
- $\theta_{j}$: the specific feature being updated
- $:=$ is assignment (in Python it's like writing `==`)
- $\alpha$: learning rate
- $m$: number of training examples
- $x^{(i)}$: the feature array of the *i*th observation
- $h_{\theta}(x^{(i)})$: returns the predicted value for the *i*th observation
- $y^{(i)}$: the observed value for the *i*th observation

A vectorized version as written in [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) is:

$\theta^{\text{next step}} = \theta - \eta \frac{2}{m}X^{T}(X \theta -y)$

Where
- $\theta$: the theta array
- $\eta$: the learning rate (previously notated as $\alpha$); the symbol is eta
- $X$: input feature matrix
- $X^{T}$: the transpose of $X$
- $y$: the array of target values

$\frac{1}{m}$ is used to calculate the mean. So why does this equation use $\frac{2}{m}$ instead? I have no idea. Andrew Ng used $\frac{1}{2m}$ (which is just $\frac{1}{2} \times \frac{1}{m}$) to "make the math easier," so it could be something similar. Regardless, the equation converges to the same answer, because the answers are proportional and will find the same minimum. However, I expect it does effect the step size.

Why does this equation use $\eta$ (eta) instead of $\alpha$ (alpha)? Beats me. [Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/) and [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) don't even write the base linear regression equation the same, replacing $\theta$ with $\beta$, adding an error term $\epsilon$, and using $p$ to indicate the number of training examples:

$Y = \beta_{0} + \beta_{1}X_{1} + \beta_{2}X_{2} + \dotsb + \beta_{p}X_{p} + \epsilon$

Why do I find four different equations representing the same basic thing in four different books/courses on the same topic? As an English major I'm inclined to blame it on a propensity of mathematicians to confuse the rest of us. Of course, I'm sure mathematicians would be as quick to point out my inclination toward verbosity.

I go into much more detail on Gradient Descent in the <font color='red'>next, next entry</font>.

### Normal equation vs gradient descent

<table align='left'>
    <tr>
        <td><b>Comparison category</b></td>
        <td><b>Normal Equation</b></td>
        <td><b>Gradient Descent</b></td>
    </tr>
    <tr>
        <td>Alpha</td>
        <td>No need to choose alpha</td>
        <td>Need to choose alpha</td>
    </tr>
    <tr>
        <td>Interation</td>
        <td>No need to iterate</td>
        <td>Needs many iterations</td>
    </tr>
    <tr>
        <td>Computational complexity</td>
        <td>$O(n^{3})$ (need to calculate inverse of $X^{T}X$) *</td>
        <td>$O(kn^{2})$</td>
    </tr>
    <tr>
        <td>Speed with large feature set</td>
        <td>Slow if <i>n</i> is very large</td>
        <td>Works well when <i>n</i> is large</td>
    </tr></table>

\* Scikit-learn implementation of OLS uses psudoinverse instead of inverse, resolving this limitation.

Where:

- $n$: number of features
- $X$: feature matrix

## Parameters

There are no tuning parameters.

## Strengths

- Fast to train
- Fast to predict
  - Computational complexity is linear
  - Ex: it takes twice as long to predict on twice as many instances (or twice as many features)
- Easily scales to very large datasets
- Works well with sparse data
- Easy to intrepret / easy to see feature importance
- Performs well when the number of features is large compared to the number of observations (ex, 104 features but only 5 observations)
- Minimizes bias

## Limitations

- In low dimensions, linear models appear to have very limited usefulness. However, as more dimensions are added, the model becomes more powerful and can become overfit
- Often unclear why coefficients are the what they are, particularly if there are highly correlated features
- Features should be scaled to improve the algorithms ability/speed to converge on the correct solution (if you've forgotten what centering and scaling are, see [Entry 8](https://julielinx.github.io/blog/08_center_scale_and_latex/))
- Specializes in linear relationships
  - While features can be augmented to help capture curvilinear relationships (like quadratic or cubic), Linear Regression may not adequately capture nonlinear relationships
  - Adding additional features to augment curvilinear relationships can create or exacerbate model overfitting
- As it uses the mean of the residuals, it is susceptible to outliers

#### Computational complexity

A few notes on computational complexity. The formulas in the table above are from **Reading: Normal Equation** in week 2 of the *Machine Learning course*. On that slide, he also notes that:

> [...] if we have a very large number of features, the normal equation will be slow. In practice, when *n* exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

*Hands-On Machine Learning with Scikit-Learn* adds the following in relation to feature size and computational complexity:

> Both the Normal Equation and the SVD approach get very slow when the number of features grows large (e.g., 100,000). On the positive side, both are linear with regard to the number of instances in the training set (they are both *O(m)*), so they handle large training sets efficiently, provided they can fit in memory.

It also adds that the computational complexity for the SVD implementation of the Normal Equation in `sklearn.linear_model.LinearRegression` is $O(n^{2})$. This puts it at roughly the same computational complexity as Gradient Descent. However, everything still has to fit into memory and *Hands-On Machine Learning with Scikit-Learn* purports on page 122 that Gradient Descent is much faster than Normal Equation or SVD when there are hundreds of thousands of features. Gradent Descent is still a better choice for large datasets due to these two properties.

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Machine Learning course](https://www.coursera.org/learn/machine-learning)
- [Introduction to Statistical Learning](http://faculty.marshall.usc.edu/gareth-james/ISL/)
- [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
