---
title: "Entry 14: Encoding Categorical Variables"
categories:
  - Blog
tags:
  - pre-process
  - cat encoding
  - dataset mushrooms
  - machine learning
---

## The Problem

Most machine learning algorithms require features to be numeric. Per usual, decision trees/random forests are the exception (the algorithm is just more forgiving in general). Last time I played with R, categorical variables were allowed to remain categorical for decision trees/random forests.

My tool of choice, Scikit-Learn, [doesn't allow for categoricals](https://scikit-learn.org/stable/faq.html#why-do-categorical-variables-need-preprocessing-in-scikit-learn-compared-to-other-tools). All features must be encoded as numeric values. The reasons for this have to do with the [extensive amount of work](https://scikit-learn.org/stable/faq.html#why-does-scikit-learn-not-directly-work-with-for-example-pandas-dataframe) needed to support categorical types.

The notebook where I did my code for this entry can be found on my github page in the [Entry 14 notebook](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/14_nb_encoding_cats.ipynb).

## The Options

- Scikit-Learn's `preprocessing` module
  - `Binarizer`
  - `LabelBinarizer`
  - `LabelEncoder`
  - `OneHotEncoder`
  - `OrdinalEncoder`
  - `label_binarize`
- Scikit-Learn's `feature_extractor` module
  - `DictVectorizer`
  - `FeatureHasher`
- `category-encoders`
  - Backward Difference Contrast
  - BaseN
  - Binary
  - Count
  - Hashing
  - Helmert Contrast
  - James-Stein Estimator
  - LeaveOneOut
  - M-estimator
  - Ordinal
  - One-Hot
  - Polynomial Contrast
  - Sum Contrast
  - Target Encoding
  - Weight of Evidence
- pandas
  - `.astype('category')` method + `.cat.codes` method
  - `.get_dummies()`
  - `.replace()` method + dictionary mapper

## The Proposed Solution

The `category-encoders` module appeals to me. Benefits include:
- Fully compatible with Scikit-Learn's transformers (it can be included in pipelines).
- First-class support for pandas dataframes as an input (and optionally as output).
- Can explicitly configure which columns in the data are encoded by name or index, or infer non-numeric columns regardless of input type.
- Portability: train a transformer on data, pickle it, reuse it later and get the same thing out.
- All methods are imported in one library.
- Largest number of encoding options from the three module choices.
- The BaseN option allows for multiple encoding methods which expands encoding to a tunable hyperparameter.
- Syntax very similar, or almost exact, between different methods. This allows for quickly iterating through different encoding methods to determine what works best for the specific dataset.

Compared with **Scikit-Learn** where:
- Each encoding method has to be individually imported by name from the `preprocessing` module.
- No explicit configuration of which columns to encode (it assumes all columns passed to it are categorical).

The **pandas** options are rather limited - only three methods. I used the `.astype()` method way back in [Entry 5](https://julielinx.github.io/blog/05_EDA/). From that experience these methods of transformation seem to require more code than the other options.

## The Fail

#### Pandas

I had to resort to for loops to apply the pandas methods to multiple columns. For loops tend to be slow and performance would suffer if I had to use these on a large dataset.

#### Scikit-Learn

I couldn't figure out how to return the `LabelBinarize` results as a dataframe. I finally gave up because there's an easier way to do it using the `category-encoders` package. Scikit-Learn tends to return things as arrays and the categorical encoding is no exception.

These methods also tend to assume you'll be doing a single column or every column. In my experience, most datasets are a combination of numerical and categorical features. It seems like Scikit-Learn would have solved this issue in a more intuitive way.

#### Category Encoders

This option was surprisingly intuitive. It returns a dataframe as the result so I can continue doing transformations and easily see what I'm doing. The syntax is almost exactly the same from one method to another, so it would be easy to write a function to swap out the different methods and see what performs best on what kinds of datasets (which is in fact exactly what I did by the time I got to the [Entry 15b notebook](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/15b_nb_cat_corr.ipynb)). It also has the most encoding options out of the three packages.

I suppose my only fail with this package is that I didn't write a function to just swap out the different encoders. I could also count the last entry's fail in that I don't know what at least half of these encoders actually do. But that's a problem for another series of entries.

**Side note**, this package imputes missing values by default. My example dataset didn't include any missing values, but this is something to keep in mind for the future.

## Up Next

[Encoding categoricals - implementation](https://julielinx.github.io/blog/15_cat_corr/)

## Resources
- [Category Encoders](http://contrib.scikit-learn.org/categorical-encoding/index.html)
- [category-encoders on pypi](https://pypi.org/project/category-encoders/)
- [Beyond One Hot: an exploration of categorical variables](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
- [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html)
- [One-Hot Encoding in Scikit-learn](https://www.ritchieng.com/machinelearning-one-hot-encoding/)
- [Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)
