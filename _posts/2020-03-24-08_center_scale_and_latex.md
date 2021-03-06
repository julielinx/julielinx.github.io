---
title: "Entry 8: Centering and Scaling"
categories:
  - Blog
tags:
  - pre-process
  - latex
  - dataset planets
  - machine learning
---

By the end of [Entry 7](https://julielinx.github.io/blog/07_collinearity/) I'd finalized the feature set to train a model and make predictions. One last step is needed before a prediction can be made.

The notebook where I did my code for this entry can be found on my github page in the [Entry 8 notebook](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/08_nb_center_scale.ipynb)

## The Problem

According to *[Hands-On Machine Learning with Scikit-Learn & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)* by Aurelien Geron: "With few exceptions, Machine Learning algorithms don't perform well when the input numerical attributes have very different scales."

Algorithms like linear regression prefer normalized and standardized values. So when the range of values (the maximum value minus the minimum value: max - min) in one column is less than 23 (`escape_vel_km_s`) and the range in another column is more than 140,000 (`diameter_km`), the algorithm takes longer to reach an answer and will probably be biased toward features with larger ranges.

There are various ways to deal with this and the terms quickly become confusing: centering, scaling, normalization, and standardization.

## The Options

According to Andrew Ng in the [Machine Learning](https://www.coursera.org/learn/machine-learning/) course by Stanford on Coursera, feature scaling and normalization can be defined as follows:

**Feature scaling**: dividing the input values by the range of the input variable, resulting in a new range of 1. Basically, you bring all the features within the same range of values.

Example: For `diameter_km` the minimum is 2,370 and the maximum is 142,984. To scale Mercury's diameter of 4,879:
- Mercury's diameter = 4,879
- `diameter_km` range = 142,984 - 2,370 = 140,614
- standardized value = $\frac{4879}{140614} = 0.0347$

**Mean normalization**: subtracting the average for an input variable from the value for the input variable, resulting in a new average value for the input variable of zero. I.E., this resets the mean value of the feature to 0.

Example: To normalize the `mass_1024_kg` of Mercury:
- Mercury's mass = 0.33
- `mass_1024kg` mean = 242.438
- normalized value = 0.330 - 242.438 = -242.108

### More definitions

Cross referencing this with *Hands-On Machine Learning* and *[Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn/dp/1461468485)* by Max Kuhn and Kjell Johnson produced some more helpful definitions and equations.

#### Math symbols

First, some helpful references:

- $\mu$ = mean
- $\sigma$ = standard deviation
- $z$ = z-score
- $X$ = a list (or column) of values
- $x$ = a specific value within $X$

**Centering**

*Applied Predictive Modeling*'s definition: "the average predictor value is subtracted from all the values."

This transformation results in a new mean of 0. Mathematically it can be represented with the following equation:

$x_{cen} = x - \mu$

If you're paying attention, this is the same thing as mean normalization above.

**Scaling**

Scaling transformations put all the features on the same scale, usually 0 to 1 or -1 to 1.

This can be done via normalization (dividing by the range like I did in the Feature Scaling definition) or standardization (dividing by the standard deviation). In addition to making the features easier for the machine learning algorithms to use, scaling can also allow dissimilar features to be compared.

To see this in action, we can look at comparing diameter and gravity. Where as comparing Mercury's diameter of 4,879 to its gravity of 3.7 isn't particularly enlightening, comparing the mean normalized scaled features of diameter 0.0347 and gravity 0.165 shows that Mercury is on the low end for both features.

**Min-max scaling (normalization)**

Geron's definition: "values are shifted and rescaled so that they end up ranging from 0 to 1."

Based on his explanation, the important part of the equation is the denominator - dividing by the range. The numerator can be the raw value, the centered value, or the value minus the min. As such, the equations for min-max scaling include:

$\displaystyle x_{scale} = \frac{x}{X_{max}-X_{min}}$

$\displaystyle x_{minscale} = \frac{x-x_{min}}{X_{max}-X_{min}}$

$\displaystyle x_{norm} = \frac{x-\mu}{X_{max}-X_{min}}$

Based on my reading, normalization is usually defined as the last equation - the centered value divided by the range.

**Standardization**

Interestingly, *Hands-On Machine Learning* and *Applied Predictive Modeling* differ in their definitions of standardization. *Hands-On Machine Learning* says to divide by the variance. *Applied Predictive Modeling* says to divide by the standard deviation. Variance ($\sigma^2$) is just the square of standard deviation ($\sigma$).

I'm going to use Andrew Ng as the tie breaker. He uses standard deviation for standardization. As such, the equation for standardization is:

$\displaystyle x_{stand} = \frac{x - \mu}{\sigma}$

The benefit of standardization is that the value is centered with a mean of 0 and a standard deviation of 1. The reason this is good is that the standardized value is less affected by outliers, making it more robust.

*Note*: standardization doesn't bound values to a specific range the way normalization does, i.e., the range of values can be outside 0 to 1.

**Downside**

Per *Applied Predictive Modeling* "the only real downside to these transformations is a loss of interpretability of the individual values since the data are no longer in the original units." However, as discussed below in The Fail section the transformation parameters can be saved, which would allow the transformation to be reversed.

## The Proposed Solution

The features in the planets dataset certainly have a wide variety of ranges. As such, normalization or standardization are recommended. Based on the EDA completed in [Entry 5](https://julielinx.github.io/blog/05_EDA/), Jupiter tends to be an outlier in several of the features. Based on this, I applied standardization to all the features.
 
 So, the visualization step was useful. I wouldn't have known about the outlier if I'd only done the automated correlation/collinear step - which is a good argument for mathematically determining skewness and outliers if the visualization step is eliminated.

## The Fail

### Time

Usually I work on these entries over a couple of sessions as I have time. However, for this entry I sat down and did it all in one go. I assumed it wouldn't take long. I already had the basic code to run the standardization and the dataset had already been defined, so the code itself only took about 20 minutes. The write up, however, took over 3.5 hours.

While this is a fail in that it took longer than the goal of 1-2 hours to complete, it also highlights the importance of breaking the problem out into the smallest level task possible. There are always unexpected things that pop up or tasks that take longer than anticipated.

Just getting the equations and mathematical symbols into the entry took about a half hour of research on LaTeX. And while I thought I understood normalization and standardization, once I tried to write down the definitions and confirm, I realized that I was actually wrong and that there are four terms for what I thought were two concepts.

This also emphasizes the importance of the write up. If I hadn't done the write up with the equations and definitions, I would have continued thinking I understood when there was actually a hole in my knowledge base.

### Reproducibility - edit

I didn't realize this until I was trying to transform the 'test' data, but the same parameters used to transform the training data need to be used on the test data. Academically, I knew I'd need to transform the test data, but the practical task of actually retaining the parameters completely slipped my mind.

The transformation parameters as applied to the training data can be saved to be applied to the test data. Since I didn't notice the problem until I needed to transform the test data and this entry was already complete, I'm going to leave this entry the way it is and change the code for the functions that retain the transformation information in the notebook where I noticed this was a problem. See [Entry 10](https://julielinx.github.io/blog/10_reorder_and_predict/) for the corrected functions.

## Up Next

[Train a model](https://julielinx.github.io/blog/09_train_model/)

## Resources

For more on scaling and centering see:
- [About Feature Scaling and Normalization – and the effect of standardization for machine learning algorithms](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
- [How, When and Why Should You Normalize / Standardize / Rescale Your Data?](https://medium.com/@swethalakshmanan14/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff)
- [Hands-On Machine Learning with Scikit-Learn & TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1491962291)
- [Machine Learning](https://www.coursera.org/learn/machine-learning/)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn/dp/1461468485)

LaTeX/Mathematics is available in Jupyter Notebook thanks to MathJax, which is included as part of the notebook functionality (no library import needed). Some references on how to use LaTex in Jupyter are:

- [Wikibooks](https://en.wikibooks.org/wiki/LaTeX/Mathematics#List_of_mathematical_symbols): lists of symbols, operators, and examples
- [Motivating Examples](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html): Examples from Jupyter's readthedocs
- [Spacing in math mode](https://www.overleaf.com/learn/latex/Spacing_in_math_mode): how to add spaces in equations
- [Advanced Jupyter Notebooks](https://blog.dominodatalab.com/lesser-known-ways-of-using-notebooks/): includes a lot of Magics functions, including %%latex