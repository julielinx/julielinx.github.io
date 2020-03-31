# Entry 15 - Categorical Correlation/Collinearity


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

raw = pd.read_csv('../data/bikeshare.csv', usecols = ['datetime', 'count'])
raw['datetime'] = pd.to_datetime(raw['datetime'])
raw['date'] = raw['datetime'].dt.date
raw['time'] = raw['datetime'].dt.hour
# .astype('str').str.split(':', expand=True)[0].astype('int32')
raw['index'] = raw['datetime']
raw.set_index('index', inplace=True)
raw['time_of_day'] = np.nan
raw.loc[(raw['time'] >= 6) & (raw['time'] < 12), 'time_of_day'] = 'morning'
raw.loc[(raw['time'] >= 12) & (raw['time'] < 18), 'time_of_day'] = 'afternoon'
raw.loc[(raw['time'] >= 18) & (raw['time'] < 24), 'time_of_day'] = 'evening'
raw.loc[(raw['time'] >= 0) & (raw['time'] < 6), 'time_of_day'] = 'night'
raw['season'] = (raw['datetime'].dt.month%12 + 3)//3
feb_1 = raw.loc['2011-02-01 00:00:00':'2011-02-02 00:00:00']
```

## The Problem

Correlation and collinearity calculations tend to rely on numeric values or at the very least, ordinal categories where the order in which they are placed have some kind of meaning. A simple example of the kinds of challenges introduced with unordered categorical variables can be seen by looking at bikeshare by hour.

When looking at the count of bikes used in an hour, there is a clear relationship between time of day and the number of bikes shared. Most prominent are the spikes around the start and end of the work day.


```python
feb_1.plot(x='datetime', y='count', figsize=(12, 6), kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a2122d890>




![png](output_3_1.png)


When these same values are reordered by a category like morning, evening, and night, the patterns that were obvious before become obscured. And if we sort the values differently (below has a secondary sort on 'count'), it could seem like there are different relationships. The secondary sort on count gives the impression of escalation across the values. But if it had a random secondary count, there probably wouldn't be any discernable pattern.


```python
feb_1.sort_values(['time_of_day', 'count']).plot(x='time_of_day', y='count', figsize=(12, 6), kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1e809810>




![png](output_5_1.png)


When the categories aren't ordinal, doing mathematics on them becomes arbitrary and subjective - how would you subtract California from the average of states? What would the average of states even be? Kansas?

## The Options

### Encode and use correlation

I could one-hot encode the values first, then run correlation. This quickly becomes unwieldy. The 22 features of the [Mushroom Dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom) for example turn into 112 one-hot encoded features.

### Cramer's V

Based on a nomial variation of Pearson's Chi-Square Test. Has a range of [0,1], where 0 means no association and 1 is full association (no negative values, either there is an association or there isn't). Like correlation, it is symmetrical (the diagnal divides mirror images of the two sides).

### Theil's U / Uncertainity Coefficient

Because Cramer's V is symmetrical, there is potential information loss. Shaked Zychlinski provided the following illustration of this problem in [The Search for Categorical Correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9):

<img src='https://miro.medium.com/max/500/1*3Mx7I537OnQybSOMPvgqEw.png'>

If I know *x* I can't predict *y* with any certainity. However, if I know *y*, then I can accurately determine the value of *x*.

Per Shaked, Theil's U is 'based on the conditional entropy between x and y — or in human language, given the value of x, how many possible states does y have, and how often do they occur.'

The range of values is the same as Cramer's V: [0,1], where 0 means no association and 1 is full association.

## The Proposed Solution

I visualized the Spearman and Pearson correlations for all categorical encodings available via the category-encoder package. This was partly a continuation/combination of the correlation exploration from <font color='red'>Entry 7</font> and the categorical encodings from entry <font color='red'>Entry 14</font>. This also gave me something to compare to the Cramer's V and Theil's U methods.

I ran the above on four datasets:
- [Mushroom Dataset](http://archive.ics.uci.edu/ml/datasets/Mushroom)
  - Observations: 8,124
  - Attributes: 22
- [Solar Flare Dataset](http://archive.ics.uci.edu/ml/datasets/Solar+Flare)
  - Observations: 1,389
  - Attributes: 10
- [Nursery Dataset](http://archive.ics.uci.edu/ml/datasets/Nursery)
  - Observations: 12,960
  - Attributes: 8
- [Chess Dataset](http://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King-Pawn%29)
  - Observations: 3,196
  - Attributes: 36

### Expanded feature sets

In the Mushroom dataset, most of the correlation methods found gill attachment and veil color as correlated. When the categorical features get expanded out to one feature per category (ie. a color column turns into one column each for red, blue, green, etc) it gets difficult to tell if a column should be removed for collinearity. If a single category is correlated to another single category, should one full attribute be removed from the dataset? Just one of the categories?

### Mathematical relationships

I'm also not sure how well algorithms are at finding relationships between features. For example, when I was playing with the planet dataset I realized that five of the features are mathematically related. With mass, radius, volume, density, and gravity if you have any two of the values you can calculate the remaining three. Would a machine learning algorithm be able to detect and use these mathematical formulas? A question to answer once I have the evaluation suite I'll be developing in the next series of entries.

### Correlation methods

It was interesting to see the differences between the Pearson and Spearman correlations in the different datasets. For example, in the Mushroom dataset, they find almost the exact same correlations with the JamesSteinEncoder, but Spearman finds negative correlations that Spearman doesn't with the LeaveOneOutEncoder. The CatBoostEncoder also returned different results than the other encoding methods on at least two of the datasets.

Overall for this first analysis, I liked the Theil's U best. Theil's U and Cramer's V both made if very clear which categorical features were correlated, but Theil's U also indicated which of the two features should be kept.

## The Fail

I detoured a bit when visualizing each of the categorical encodings by Pearson and Spearman correlation measures. The detour did give me a much better understanding of the category-encoders package. For example, the encoding methods break down into two main categories:
- **Target independent** (the encoding method is not related to the attribute being predicted)
  - OrdinalEncoder
  - OneHotEncoder
  - BinaryEncoder
  - HashingEncoder
  - BackwardDifferenceEncoder,
  - HelmertEncoder
  - PolynomialEncoder,
  - SumEncoder
- **Target dependent** (ie. target encoded, where the encoding method is based in some way on the attribute being predicted)
  - CatBoostEncoder
  - JamesSteinEncoder
  - LeaveOneOutEncoder
  - MEstimateEncoder (likelihood - logistic predictions) and TargetEncoder (continuous variable - regression predictions)
  - WOEEncoder

The inconsistent results returned by the CatBoostEncoder will need to be explored more later. I'll also need to look into the questions raised in the Proposed Solution section: how do the algorithms understand a feature when it is expanded out and can the algorithms find and use mathematical relationships across multiple features.

The detour into Pearson and Spearman correlations also allowed me to develop a function to remove undifferentiated columns. These are columns with only a single value. Because every value is the same, these columns don't contribute to correctly predicting the target variable. In addition, correlations return errors (result is NA, which cannot have further calculations run and cannot be visualized) for these values.

## Up Next

Develop an evaluation suite.

### Resources

The major resource for this entry was [The Search for Categorical Correlation](https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9).


```python

```
