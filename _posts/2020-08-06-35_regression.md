---
title: "Entry 35: Regression"
categories:
  - Blog
tags:
  - regression
  - supervised learning
---

Regression is used to predict on continuous data, for example housing prices. Logistic Regression is the subcategory that is the exception, as it does classification only.

Scikit-Learn has parameters that allow several of the subcategories of regression, such as Ridge, Lasso, and Elastic Net, to handle both regression and classification.

The notebooks where I did my code for this entry can be found on my github page:

- [Entry 35a notebook](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/35a_nb_regression.ipynb)
- [Entry 35b notebook](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/35b_nb_regression.ipynb)

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
    <tr>
        <td></td>
        <td>Classification</td>
    </tr>
</table>

## Description

The gist of regression is that it uses a linear function to fit the data. For the simplest cases, it's just like the equation for a line:

y = mx + b

Where:

- y = y
- m = slope of the line
- x = x
- b = y-intercept

![Wave line](https://julielinx.github.io/assets/images/35a_wave_line.png)

To account for more than one input, the full equation for linear regression is a little more complicated.

The equation as listed on page 47 of [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) is:

$\hat{y} = w_{0} x_{0} + w_{1} x_{1} + \dotsb + w_{p} x_{p} + b$

Where each *x* represents a feature (just like in the single variable version where x is the input and y is the output) and *w* is a learned weight for each feature. The text continues, and points out that the equation for a single feature is:

$\hat{y} = w_{0} x_{0} + b$

This equation is exactly the same as the one for a line above, except $w_{0}$ has been substituted for *m*. A simplified way of thinking about this would be that the slope (*m*) is calculated for each feature and used as the weight (*w*). *Introduction to Machine Learning with Python* puts it like this:

> For more features, *w* contains the slopes along each feature axis. Alternatively, you can think of the predicted response as being a weighted sum of the input features, with weights (which can be negative) given by the entries of *w*.

It expounds further on page 48:

> Linear models for regression can be characterized as regression models for which the prediction is a line for a single feature, a plane when using two features, or a hyperplane in higher dimensions.

### Subcategories

So, all of that is well and good, but how are the weights/theta values calculated? According to *Introduction to Machine Learning with Python* on page 49, the way the model learns the weights and how it controls complexity are the differentiating factors between the various subcategories of regression. These subcategories are:

- Ordinary Least Squares (OLS)
  - Normal Equation
  - Gradient Descent
- Ridge Regression
- Lasso Regression
- Elastic Net
- Logistic Regression

Each subcategory will have its own entry explaining how it differs from the others.

## Purpose

The purpose of regression is to find the best fit "line" for the feature space and then apply that to data points that haven't been seen before to predict the output of the new data. The way this is implemented is via linear algebra using a vectorized version of the equation.

### Vectorized equation

The equation for linear regression as listed on page 112 of [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) looks a little different than the one in *Introduction to Machine Learning with Python*:

$\hat{y} = \theta_{0} + \theta_{1} x_{1} + \theta_{2} x_{2} + \dotsb + \theta_{n} x_{n}$

However, all the changes are superficial:
- *b* is moved to the front of the equation and written as $\theta_{0}$
- The *w*s are written as $\theta$s
- The unknown number of elements is written as *n* instead of *p*
- The subscripts for the *x*s starts at 1 instead of 0

The nice thing about the slight alterations to the equation is that it makes it easier to understand the vectorized representation of the equation (from page 113 of *Hands on Machine Learning with Scikit-Learn*):

$\hat{y} = h_{0}(x) = \theta x$

Where

- $\theta$ is a vector (i.e. list) of weights, with the first value being the y-intercept type value (represented by *b* in *Introduction to Machine Learning*)
- *x* is the matrix of feature values (i.e. the DataFrame) with the first column ($x_{0}$, not listed in the equation) being all 1s so that $\theta_{0}$ is always evaluated as the same value

As an example, I created a fake theta array for a subset of the planet data. The first value in the theta array (6.5) is *b* in *Introduction to Machine Learning with Python* and $\theta_{0}$ in *Hands-On Machine Learning with Scikit-Learn*. The rest of the values are the weights for each feature (the *w*s in *Introduction to Machine Learning with Python* and $\theta_{1}$ through $\theta_{n}$ in *Hands-On Machine Learning with Scikit-Learn*.


```python
theta = pd.Series([6.5, 2.5, 8.1, 0.3, 1.7, 3.8, 5.9])
theta
```

    0    6.5
    1    2.5
    2    8.1
    3    0.3
    4    1.7
    5    3.8
    6    5.9
    dtype: float64

The subset of planet data, along with the initial column of 1s (remember the first column needs to be all 1s so that $\theta_{0}$ remains the same value) looks like this:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>theta0</th>
      <th>mass_1024kg</th>
      <th>diameter_km</th>
      <th>mean_radius_km</th>
      <th>density_kg_m3</th>
      <th>gravity_m_s2</th>
      <th>escape_vel_km_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.3300</td>
      <td>4879.0</td>
      <td>2439.4000</td>
      <td>5427</td>
      <td>3.7</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>4.8700</td>
      <td>12104.0</td>
      <td>6051.8000</td>
      <td>5243</td>
      <td>8.9</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.9700</td>
      <td>12756.0</td>
      <td>6371.0084</td>
      <td>5514</td>
      <td>9.8</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.0730</td>
      <td>3475.0</td>
      <td>1737.4000</td>
      <td>3340</td>
      <td>1.6</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.6420</td>
      <td>6792.0</td>
      <td>3389.5000</td>
      <td>3933</td>
      <td>3.7</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1898.0000</td>
      <td>142984.0</td>
      <td>69911.0000</td>
      <td>1326</td>
      <td>23.1</td>
      <td>59.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>568.0000</td>
      <td>120536.0</td>
      <td>58232.0000</td>
      <td>687</td>
      <td>9.0</td>
      <td>35.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0.1260</td>
      <td>5149.4</td>
      <td>2574.7000</td>
      <td>1882</td>
      <td>1.4</td>
      <td>2.6</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>86.8000</td>
      <td>51118.0</td>
      <td>25362.0000</td>
      <td>1271</td>
      <td>8.7</td>
      <td>21.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>102.0000</td>
      <td>49528.0</td>
      <td>24622.0000</td>
      <td>1638</td>
      <td>11.0</td>
      <td>23.5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0.0146</td>
      <td>2370.0</td>
      <td>1188.3000</td>
      <td>2095</td>
      <td>0.7</td>
      <td>1.3</td>
    </tr>
  </tbody>
</table>
</div>



The dot product is then calculated for the theta array and the planet feature values to give the linear regression output.

```python
planet_df.values.dot(theta)
```

    array([  49524.37497204,  108884.89493742,  114733.44748931,
             34373.64250797,   62769.81500023, 1186588.23000145,
            996649.25      ,   45709.42419369,  424207.3300024 ,
            411789.95      ,   23131.85651432])


For more information on linear algebra/matrix multiplication and linear regression, see weeks 1 and 2 of Andrew Ng's [Machine Learning](https://www.coursera.org/learn/machine-learning) course on [coursera.org](https://www.coursera.org/). For information on how to implement it in a machine learning context, see the note in *Hands-On Machine Learning with Scikit-Learn* on page 113 (mainly when and why $\theta$ needs to be transposed).

## Behavior

### Single feature

The line for a single feature is easily visualized, just as in the first chart at the beginning of the entry, which uses the synthetic Wave dataset created for the book. Another example can be seen using the `car_crashes` sample dataset in Seaborn:


![Wave line](https://julielinx.github.io/assets/images/35b_car_alcohol_line.png)


### Two Features

The plane for two features is also easily visualized in a 3D visualization.

The concept of a plane is easily seen in this chart from the blog [Machine Learning in Action](https://appliedmachinelearning.blog/2017/03/09/understanding-support-vector-machines-a-primer/) during a discussion of SVMs.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/img/hyperplane2.png?raw=true'>

Taking this concept and applying it to data in a plane (instead of separated by a plane like the SVM based example above), I created a visualization based on the planet dataset I originally started this data science journaling journey with. Remember how the data didn't meet the criteria of independently and indentically distributed? Well, now that works in my favor because I know exactly which features *should* be correlated and the mathematical way in which they're related.

For data that should show on a plane, I needed a feature that depended on two other variables. My options were:

- density = $\frac{M}{v}$
- gravity = $\frac{GM}{r^2}$
- escape velocity = $\sqrt{\frac{2GM}{r}}$

Where:

- M = mass
- v = volume
- G = gravational constant
- r = radius

Of the three, density had the most obvious example:

![Density chart](https://github.com/julielinx/datascience_diaries/blob/master/img/density_3d.png?raw=true)

The color saturation of the dot indicates how close it is to the viewer (darker is closer, lighter is farther away). In this particular configuration, the data points almost seem to form a line (except one data point, which stands above the line). However, the darkess of the dots reveals that they vary along the near/far axis. As such, it appears they all (except one) fall along the same plane.

### More than Three Features

As the number of dimensions goes up, it becomes more and more difficult to visualize and conceptualize.

## Parameters

Depend on which subcategory of regression is being used. This will be populated in the entry for each specific subcategory.

## Strengths

- Fast to train
- Fast to predict
- Easily scale to very large datasets
- Work well with sparse data
- Easy to intrepret / easy to see feature importance
- Performs well when the number of features is large compared to the number of observations (ex, 104 features but only 5 observations)

## Limitations

- In low dimensions, linear models appear to have very limited usefulness. However, as more dimensions are added, the model becomes more powerful and can become overfit
- Often unclear why coefficients are the what they are, particularly if there are highly correlated features
- Specializes in linear relationships
  - While features can be augmented to help capture curvilinear relationships (like quadratic or cubic), Linear Regression may not adequately capture nonlinear relationships
  - Adding additional features to augment curvilinear relationships can create or exacerbate model overfitting
- As it uses the mean of the residuals, it is susceptible to outliers
- May become erratic when the number of predictors is higher than the number of observations (ex, 14 features but only 5 observations)

## Evaluation

The appropriate metrics depend on whether the model is being used for regression or classification. Discussions on the available metrics can be found in prior entries:

- [Entry 21 - Scoring Regression Models](https://julielinx.github.io/blog/21_reg_score_theory/)
- [Entry 23 - Scoring Classification Models](https://julielinx.github.io/blog/23_class_score_theory/)

## Datasets

Appropriate datasets depend on whether the model is being used for regression or classification. Some same datasets include:

- [Entry 34a - Regression Datasets](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/34a_nb_regression_datasets.ipynb)
- [Entry 34b - Classification Datasets](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/34b_nb_classification_datasets.ipynb)

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Machine Learning in Action](https://appliedmachinelearning.blog/2017/03/09/understanding-support-vector-machines-a-primer/)
- [Three-Dimensional Plotting in Matplotlib](https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html)
- [mplot3d tutorial](https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html)
- [Machine Learning](https://www.coursera.org/learn/machine-learning) coursera course by Andrew Ng
- [pandas matrix dot product failed for the two matrix having the same dimension - pandas](https://html.developreference.com/article/12561922/pandas+matrix+dot+product+failed+for+the+two+matrix+having+the+same+dimension)

## Up Next

[Ordinary Least Squares (OLS)]((https://julielinx.github.io/blog/36_regression_OLS/))
