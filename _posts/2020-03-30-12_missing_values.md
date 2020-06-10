---
title: "Entry 12: Missing Values"
categories:
  - Blog
tags:
  - pre-process
  - dataset csm (conventional and social media movies)
  - dataset titanic
---

[Wikipedia](https://en.wikipedia.org/wiki/Missing_data) has a succinct definition of missing values: "missing data, or missing values, occur when no data value is stored for the variable in an observation." Seems straight forward, but there are different underling reasons for missing data and those reasons can compound their effect in a model.

The notebook where I did my code for this entry can be found on my github page in the [Entry 12 notebook](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/12_nb_missing_values.ipynb)

## The Problem

Most models are unable to run missing values. Tree-based models are an exception to this rule, but many of the others are intolerant of missingness. Unless I want to be confined to tree-based models (I don't, neural nets are basically a bunch of regression models strung together, and they build some of the most robust and accurate models currently possible) I need to be able to account for missing information.

The major hurdle with this issue, as stated above, is that a missing value can mean different things in different contexts. There are various ways to indicate missingness in a dataset, some more appropriate than others in certain contexts.

### Definitions

There are [two general ways](http://www.stat.cmu.edu/~hseltman/726/Missing%20Data%20726.pdf) to define missing data:

- Definitions based on *representation*: Missing data is the lack of a recorded answer for a particular field.
  - If I received a call from a customer and the system usually logs the phone number they called from, if the logging system is down that would be a lack of a recorded answer.
  - If the customer interacts with me on the internet and the system usually logs the IP address, the phone number logging entry would be empty.
  - Most every model type except tree-based models evaluate missingness on this criteria. If the value is empty, it needs to be addressed one way or another before the model can be run.
- Definitions based on *context*: Missing data is lack of a recorded answer where we “expected” to find one.
  - In the second example above with the internet customer interaction, I wouldn't expect to record the phone number they're calling from - the value would be missing, but it's not expected in the context.
  - Context can be addressed through modeling choices - ie only model for phone interactions or web interactions (separating these two interaction types immediately reduces the number of missing values because it considers context)

Within these two definition types we can see the following types of missingness as defined in section 3.4 of [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn/dp/1461468485) by Max Kuhn and Kjell Johnson:

- **Undetermined**: the value was not determined at the time the data was gathered.
  - Example: For the same customer interaction data, if the process that captures IP addresses goes down we would be unable to capture the data - the information existed but was unable to be determined/recorded.
- **Structurally missing**: the missing value itself has information.
  - Example: For the customer interaction scenario, a missing IP for a phone interaction is structurally missing - there was no IP to be captured.

### Categories

[According to Andrew Gelman](http://www.stat.columbia.edu/~gelman/arm/missing.pdf), a statistics professor at Columbia University, in his book [Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/) misisng data is generally grouped into four categories:

- **Missing completely at random** (MCAR)
  - The probability of missingness is the same for all units
  - Example: Rows accidentally skipped during data entry
- **Missing at random** (MAR)
  - The  the probability of missingness depends on only other, fully recorded variables
  - Example: A power outage shut down a remote sensor and the power outage is captured in another variable
  - The other variables don't have to predict the missing value, just that it's missing
- **Missing not at random I** (MNAR)
  - Missingness depends on information that has not been recorded
  - Example: The same power outage scenario as in MAR, but the power outage wasn't recorded
- **Missing not at random II** (MNAR)
  - The probability of missingness depends on the variable itself
  - Example: people with higher earnings are less likely to reveal their salary
  - **Censoring**: where the exact value is missing but something is known about its value
    - Example: People who make >$100K don't report their earnings
    - Example: Lighter weight chicks die before they can be sampled
    - Example: A sensor cannot measure values under a specific value
    - Example: A duration is unknown because the interaction isn't yet complete

## The Options

There are four ways to deal with missing data:

- **Remove the observation** (delete the whole row)
- **Remove the variable** (delete the whole column)
- **Create a new category or enter a dummy value** (new category - like "missing" or "unknown"; dummy variable - like 999 or -1)
- **Impute the value** (make an educated guess)

The first three options are pretty straight forward, but the fourth has a variety of sub-options.

### Imputation

There are quite a few options when it comes to imputation. 

- Pandas' [.fillna()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.fillna.html) method allows the following [strategies](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html#filling-missing-values-fillna):
  - scalar (ie: a constant)
  - forward fill
  - backward fill
  - pandas object (ie: the mean, median, etc)
- Scikit Learn's [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) has the following options:
  - mean
  - median
  - most_frequent
  - constant
- Scikit Learn's [IterativeImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer): an experimental strategy for imputing missing values by modeling each feature as a function of other features. It has the following initial strategies:
  - mean
  - median
  - most_frequent
  - constant
- Predict the missing value using the other features as attributes

## The Proposed Solution

To explore the different methods of addressing missing data, I decided to use the [CMS Dataset](http://archive.ics.uci.edu/ml/datasets/CSM+%28Conventional+and+Social+Media+Movies%29+Dataset+2014+and+2015#) from the UCI Repository. Three of the twelve attributes had missing values.

The first three methods for addressing missing data (remove the row, remove the column, create a dummy value) were very straight forward. The most sophisticated thing I did with those was to use a threshold or subset columns.

Most of the sub-options for the fourth method (imputation) were also straight forward - basically replacing the missing value with a constant where the constant is mathematically derived. There are definitely specific sub-options that work better for certain distributions though.

- Mean is good when there is an even distribution of values.
- Median is good when there are outliers that distort the mean.
- Most frequent is best for situations where one value dominates the others (like power law distributions and highly skewed data)

## The Fail

### Accounting for distribution

I was unable to account for the [bimodal distribution](https://en.wikipedia.org/wiki/Multimodal_distribution) of one of the attributes - Screens. There were two distinct peaks in the distribution which made mean, median, and most frequent all bad choices. Mean and median would both start to fill in the dip between the peaks, while most frequent would just make the higher peak more prominent.

IterativeImputer looked promising, as it says it models each feature as a function of other features. However, when applied to the dataset, it gave the same results as using the mean or median.

The last solution, predicting the missing value using the remaining features as attributes, would essentially be a model within a model. I'm not sure how I feel about the validity of that and prefer to wait until I have a model diagnostic performance suite in place before attempting it.

### Automating the solution

The tricky part of addressing missing values will be automating the solution. Since each method is best used on a specific criteria (frequency of missingness, data distribution, etc), I'll need to develop logic and code to programmatically determine the best solution given the data's characteristics. Like feature reduction in [Entry 11](https://julielinx.github.io/blog/11_consolidate_preprocess), this gets into the territory of needing a way to measure the efficacy of my logic. All of which sounds time consuming. This was already another time-consuming, research heavy entry that took way longer than 1-2 hours. So my final decision is that I'm going to skip automating the missingness solution until I've got a way to evaluate model performance.

## Up Next

Endcoding categorical variables

## Resources

- [Overview of Approaches for Missing Data](http://www.stat.cmu.edu/~hseltman/726/Missing%20Data%20726.pdf) - powerpoint from a Carnegie Mellon course
- [The Importance of Missing Data](http://www.simonqueenborough.info/R/basic/missing-data) - high level overview of missing data
- [Missing-data imputation](http://www.stat.columbia.edu/~gelman/arm/missing.pdf) - chapter 25 of Data Analysis Using Regression and Multilevel/Hierarchical Models by Andrew Gelman and Jennifer Hill
- [Missing data](https://en.wikipedia.org/wiki/Missing_data) - Wikipedia entry
- Missingno - visualization package for missing data
  - [GitHub repo](https://github.com/ResidentMario/missingno) pip install instructions and function calls
  - [Anaconda installation](https://anaconda.org/conda-forge/missingno)
