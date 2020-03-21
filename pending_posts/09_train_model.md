# Entry 9 - Train Model

At the end of the noteboook for <font color='red'>Entry 8</font> I had a standardized dataset. Now it's time to try training a model.

## The Problem

Train a model using the standardized dataset where the collinear features have been removed.

## The Options

Scikit learn is the most common package for building models. Implementing by hand is also an option. Andrew Ng teaches manual implementation of linear and logistic regression in his [Machine Learning course](https://www.coursera.org/learn/machine-learning/) on Coursera. However, using a package will allow me to iterate through multiple types of models faster than learning how to implement and optimize each one of them by hand. As an added benefit, by using an open source package, there are a lot of people contributing to them, so the optimazation is probably better than anything I could come up with without several more years of experience under my belt.

## The Proposed Solution

As discussed in <font color='red'>Entry 2</font>, Scikit-learn lists [17 different kinds of algorithms](https://scikit-learn.org/stable/supervised_learning.html). While working through the process on this first model, I'm going to stick to one of the most basic models - linear regression.

The actual implementation of the model training went smoothly.
- The data was all ready after <font color='red'>Entry 8</font>, so no further pre-processing was required.
- Separating the target from the attributes was easily accomplished with pandas.
- Training the model was as easy as specifying the model type and calling the fit() function.

## The Fail

There were several pitfalls in this exercise.

### Missing Values

While there were no missing values in this particular, hand-curated, dataset, missing values are extremely common in the wild. I'll need to address this for any kind of standardized process or automated pipeline.

### Validating the model

This dataset is really small (only 11 observations), so splitting it into train and test sets would leave very little data to work with or to validate on.  I tried looking at the coefficients and score, but these values are currently meaningless to me.

I decided to skip this step for now. A full set of model diagnostics will be developed in a <font color='red'>future entry</font> based on week 6 of Andrew Ng's Machine Learning course and any other resources I can find.

### Retain Scaling

I had planned on making a prediction in this entry too. Unfortunately, I forgot about retaining scaling parameters during my pre-processing stage. For this particular dataset, I could get around the categorical encoding by copying the line of interest from the already encoded version of the data (I just made two copies of Mars' row then changed the surface pressure to the high and low values that humans can withstand based on survivable conditions (nigrogen narcosis in divers being the high end and conditions experienced by climbers on Mt Everest being the low end), but the categorical encoding will need to be retained for any other dataset I'd want to evaluate (I wouldn't be copying existing data - it should be data that the model has never seen before).

### Process Order

Separating the target variable after all the pre-processing leaves me with a scaled value. The scaled value has no meaning and would need to be un-scaled to get the actual prediction. In reviewing the purpose of scaling (to allow the model to quickly converge on a better solution), there really isn't a reason to scale the target value. As such, it should be separated from the features before any pre-processing.

## Next Up

Fixing Entry 9 failures and making a prediction.


```python

```
