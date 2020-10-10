---
title: "Entry 22: Scoring Regression models - Implementation"
categories:
  - Blog
tags:
  - model-eval
  - regression models
  - dataset auto mpg
---

Now that I've got a handle on the measurement options and equations, it's time to implement those measures on actual models.

The notebook where I did my code for this entry can be found on my github page in the [Entry 22 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/22a_nb_reg_score_implement.ipynb).

## The Problem

In [Entry 21](https://julielinx.github.io/blog/21_reg_score_theory/) I covered the mathematical options for measuring regression models. But just because I know the equations doesn't mean I can apply it to actual data.

## The Options

The two primary options are to list metrics in the `scoring` parameter of a function like `cross_validate` or to use a function from the `metrics` module. There are quite a few things the two methods have in common.

The equations behind the functions are the same and the function names are very similar. However, the error and deviance functions return negative values when used in the `scoring` parameter and appear to return positive values when used in the `metrics` module's functions. I only suspect this because the `metrics` functions don't have `neg_` prefixed to the function names.

In general, the naming conventions follow a few rules. According to [Scikit-Learn's documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring):

- functions ending with `_score` return a value to maximize, the higher the better
- functions ending with `_error` or `_loss` return a value to minimize, the lower the better

### `scoring` parameter

The first option is to just list scoring methods in the scoring parameter of `cross_validate`. I finally found the list of options on the [3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html) page under the [3.3.1. The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter) section, which also groups the options into either classification, clustering, or regression.

The metrics that can be used with this method are limited to those that don't require extra parameters. This standardization makes it possible to just name the metric without having to bother with dictionaries, lists, optional parameters or any other add ons. The available regression metrics are:

- `explained_variance`
- `r2`
- `max_error`
- `neg_median_absolute_error`
- `neg_mean_absolute_error`
- `neg_mean_squared_error`
- `neg_mean_squared_log_error`
- `neg_root_mean_squared_error`
- `neg_mean_poisson_deviance`
- `neg_mean_gamma_deviance`

*Side note*, the ones prefixed with `neg_` return a negative value despite most of them being an absolute or squared value, which would normally be positive. This man have something to do with the naming convention: functions ending in "error' should be minimized.

### Context

From the list of options above, I covered $R^2$, explained variance, mean absolute error, mean squared error, and root mean squared error in [Entry 21](https://julielinx.github.io/blog/21_reg_score_theory/). I'll cover the additional options here.

#### Max error

[This metric](https://scikit-learn.org/stable/modules/model_evaluation.html#max-error) is on the Scikit-Learn page. It's the maximum value of the absolute errors.

$\text{max error} = max(\lvert y_{i} - \hat{y_{i}}\rvert)$

#### Median absolute error

This is the same as MAE (mean absolute error), but uses the median instead of the mean. The benefit of using this instead of MAE is the same as the benefit of using median instead of mean: it's robust to outliers.

$median\text{ }squared\text{ }error = median(\sum \lvert y_{i} - \hat{y_{i}}\rvert)$

#### Mean squared logarithmic error

This is just the MSE with a logarithmic component. [Scikit-Learn recommends](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error) using it with targets that show exponential growth. Keep in mind however that because it uses exponentiation it penalizes under-predicted estimates more than over-predicted estimates (larger numbers are reduced more than small numbers using logarithmic functions). 

$MSLE = \frac{1}{n} \sum{(log_{e}(1+y_{i}) - log_{e}(1+\hat{y_{i}}))}^2$

#### Mean poisson deviance and mean gamma deviance

Remember back at the beginning of this section when I said only metrics that don't require extra parameters can be used with the `cross_validate` function? These two metrics are listed explicitly because of that.

Mean poisson/gamma deviance, along with MSE, belong to a function called tweedie deviance. Mean tweedie deviance error takes a `power` parameter. Power 0 = MSE, power 1 = poisson, power 2 = gamma. The higher the power the less sensitive the metric is to extreme deviations. Scikit-Learn has some good examples in the [3.3.4.8. Mean Poisson, Gamma, and Tweedie deviances](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance) section.

$mean\text{ }poisson\text{ }deviance = 2(y_{i} log(\frac{y_{i}}{\hat{y_{i}}}) + \hat{y_{i}} - y_{i})$

$mean\text{ }gamma\text{ }deviance = 2(log(\frac{\hat{y_{i}}}{y_{i}}) + \frac{y_{i}}{\hat{y_{i}}} - 1)$

### `metrics` module

The functions in the `metrics` module allow for more flexibility than the predefined options in the `scoring` parameter due to being able to take additional parameters. The `mean_squared_error`, `mean_absolute_error`, `explained_variance_score`, and `r2_score functions` can handle multi-output cases. Multi-output cases aren't something I work on very often, so I'm going to leave coverage of this topic at that. See the [Scikit-Learn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) for more information.

The only additional function available is the full tweedie deviance, which accepts different power inputs. Other than that, the only difference is that the names of the functions are slightly altered. 

- `explained_variance_score`
- `max_error`
- `mean_absolute_error`
- `mean_squared_error`
- `mean_squared_log_error`
- `median_absolute_error`
- `r2_score`
- `mean_poisson_deviance`
- `mean_gamma_deviance`
- `mean_tweedie_deviance`

The `make_scorer` function from the `metrics` module makes these functions easily accessible. As mentioned above, the functions follow a naming convention which makes it easy to use them with `make_scorer` and `cross-validate`:

- functions ending with `_score` return a value to maximize, the higher the better
- functions ending with `_error` or `_loss` return a value to minimize, the lower the better. When converting into a scorer object using `make_scorer`, set the `greater_is_better` parameter to `False`

The `make_scorer` function also allows for the creation of custom scoring functions. Details on how to do this with examples can be found in section [3.3.1.2. Defining your scoring strategy from metric functions](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring) of the Scikit-Learn documentation.

## The Proposed Solution

I list all of the regression metrics in the `scoring` parameter of the `cross_validate` function. The code is basically just a cleaned up version of [Entry 20's notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/20a_nb_sklearn_pipeline.ipynb) with all the metrics applicable to a continuous target plugged in.

The most interesting thing about this was to see the difference between the value returned by the default `score` method (0.81) vs the range of values returned from `cross_validate`. For example, max error ranged from -3.74 to -11.58, $R^2$ ranged from 0.56 to 0.94, and negative root mean squared error ranged from -2.17 to -5.32. $R^2$ and explained variance didn't match, so the mean(error) for this dataset obviously isn't 0.

The range of values makes clear the impact that the splitting of the data makes. A range from 0.56 to 0.94 is much more enlightening than a single value of 0.81.

## The Fail

After cobbling together everything for the theory entry, this one seemed downright easy. All the information I needed was in the Scikit-Learn documentation.

I had to break these two entries into separate posts due to the sheer amount of text needed to flesh out the various equations, provide context for baseline terms and definitions, and explain the nuances of the coding options.

## Up Next

[Classification metrics - theory](https://julielinx.github.io/blog/23_class_score_theory/)

### Resources

These links all lead to the same basic page: 3.3 Metrics and scoring (the first link). The other three are links to specific portions of the documentation that are relevant to regression metrics.

- [3.3. Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [3.3.1. The scoring parameter: defining model evaluation rules](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules)
- [3.3.4. Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [3.3.1.2. Defining your scoring strategy from metric functions](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring)