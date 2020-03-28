---
title: "Entry 10: Reorder Pre-processing and Make Predictions"
categories:
  - Blog
tags:
  - pre-process
---

Using the pre-processing steps I worked through in entries 6-8, I can now predict mass while only changing the surface pressure.

The notebooks where I did my code for this entry can be found on my github page in the [Entry 10a notebook (target value: mass)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/10a_nb_reorder_and_predict.ipynb) and [Entry 10b notebook (target value: atmospheric mass)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/10b_nb_reorder_and_predict.ipynb).

## The Problem

[Entry 9](https://julielinx.github.io/blog/train_model/) resulted in a trained model. However, I ran into several problems that need to be addressed before a prediction can be made. These problems are:

- Categorical and scaling parameters weren't retained, so they couldn't be applied to the test data
- Target value would be scaled during the standardization step, rendering the predictions uninterpretable

## The Options

Retaining the pre-processing transformations is the easy point to address, as there's basically only one option: retain the information so you can apply it again later. This is easily accomplished with Scikit-Learn's `preprocessing` module. I just have to return the information as part of my function.

Addressing the second point requires more thought. The target value being scaled is making me reconsider the order in which the pre-processing occurs. I could just separate the target value and features at the scaling step, but I think I should address several other concerns at this point so that the process is easier to apply to other datasets.

## The Proposed Solution

Based on the issues encountered, I propose updating my pre-processing steps to the following:

### Split dataset

The very first step in the standardized process after loading the data should be to split it into train, test, and reserve datasets.

- **Train**: where pre-processing is run, features are generated, and the model is trained
- **Test**: where the trained model(s) is/are tested on data never seen before and hyperparameters are evaluated
- **Reserve**: where the final model is assessed

### Separate target value and features

The target value doesn't need to be pre-processed (except maybe for missing values, but as I'm only dealing with supervised learning at this stage, there shouldn't be any missing values in the target). The easiest way to ensure no pre-processing transformations are applied is to split off the target from the rest of the data before pre-processing.

### Determine collinearity

Correlation doesn't care about scaling (I checked - the values I got when running correlation on the unscaled features matched the values Sabber and I got when running it on standardized values). It might only make a miniscule difference on most datasets but to speed up pre-processing, I'm going to remove collinear features before applying transformations.

Correlation only works on numeric values. I'm going to explore determining collinearity of categorical-categorical and categorical-numeric features in <font color='red'>Entry 15</font>. For now, I'm going to run collinearity on just the numeric features (due to the transformation issues listed in the 'Apply transformations' section).

### Apply transformations

This is where I encode categorical features and scale numeric features. To ensure the categorical features don't accidentally get scaled (which they did in the [entry 8 and 9 notebooks](https://github.com/julielinx/datascience_diaries/tree/master/01_ml_process)), scaling should happen first on just numeric features, then categorical features can be encoded. Per [these](https://stats.stackexchange.com/questions/169350/centering-and-scaling-dummy-variables) two [posts](https://en.wikipedia.org/wiki/Categorical_variable), categorical features (including 1/0 encoding) should never be scaled.

When scaling the numeric features, one of the books/tutorials/blogs recommended scaling when the values were different by an order of magnitude (power of 10). In [Entry 8](https://julielinx.github.io/blog/center_scale_and_latex/) I delved into centering and scaling. The two most common options in scaling are normalization (centering and dividing by range) and standardization (centering and dividing by standard deviation). Normalization brings the mean to 0 and the range of values between 0 and 1. Standardization brings the mean to 0 and the standard deviation to 1. There is no set range on the standardized value, but because of [the way standard deviation works](https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data), for normally distributed data 68.3% of values should be within 1 standard deviation, 95.4% within 2 standard deviations, and 99.7% within 3 standard deviations. As such, values an order of magnitude larger between features should be rare.

If there are more than 100 categories in a categorical feature, that could present a problem. However, I plan to delve further into categorical features and the options for encoding then in <font color='red'>Entry 14</font>.

### Make predictions

Once the above steps have been implemented in order, I can finally make predictions.

## The Fail

### The predictions

I used the pipeline to predict both mass and atmospheric mass where I only varied the surface pressure in the 'test' data. At first glance, the results don't make any sense - both sets of predictions returned negative values. In attempting to raise the surface pressure of Mars, I expected positive values for both predicted mass and atmospheric mass. Upon further reflection, I was able to make sense of the mass prediction.

Changes in mass effect many of the other variables, such as diameter, gravity, density, escape velocity, etc via known mathematical equations (see [Entry 7](https://julielinx.github.io/blog/collinearity/) for the escape velocity equation). By leaving these values all the same and only changing surface pressure there would need to be more atmosphere per surface area. As atmospheric mass was removed from the features by collinearity, I expected the mass to go up. However, escape velocity and density stayed the same, putting a constraint on mass. A decrease in mass (and thus diameter - also removed by collinearity) would then allow more atmosphere per surface area.

Also, there's a statistical principle that I forgot in generating my dataset. Generally, there's an assumption of independent and identically distributed (IID or i.i.d.) variables in machine learning. IID is an important concept. [Wikipedia](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) has a good definition: 

> In probability theory and statistics, a collection of random variables is independent and identically distributed if each random variable has the same probability distribution as the others and all are mutually independent.[...] In machine learning theory, i.i.d. assumption is often made for training datasets to imply that all samples stem from the same generative process and that the generative process is assumed to have no memory of past generated samples.

The data I'm using is a major violation of the independent assumption (the values of some features are literally derived by using other variables in an equation).

My non-sense results support the adage '[garbage in garbage out](https://en.wikipedia.org/wiki/Garbage_in,_garbage_out).' I totally earned this unusable result by being lazy in changing only one value while making up my own test data and being lazy about dependent features.
 
 I could address the first problem by entering the mathematical equations so that when one value changes, the others update accordingly. However, that still won't address my IID problem.
 
 The IID assumption feels counter intuitive. If all features are independent, then no one feature could predict any of the others. Often in the real world, many variables effect each other, just like with my planets dataset.
 
 Taking into account the dependent nature of a system is probably getting into complexity theory. Based on my 1-2 hour per entry goal and current skill level, complexity theory is definitely biting off more than I can chew. For now, it's just food for thought.

### Time

The amount of time dedicated to these changes was also a complete failure from the 1-2 hour goal perspective. There were quite a few major revisions for this entry, which maybe should have been split out and addressed individually. 

However, this entry resulted in a working pipeline to go from loading data to making a prediction. Some additional automation will need to be completed around splitting the train/test data, creating the model/making predictions, and addressing missing values, but overall a lot of progress was made here.

## Up Next

[Consolidate process to date](https://julielinx.github.io/blog/consolidate_preprocess/)