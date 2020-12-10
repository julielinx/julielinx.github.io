---
title: "Entry 49: Decision Trees Analysis and Interpretation"
categories:
  - Blog
tags:
  - trees
  - supervised learning
  - dataset titanic
  - dataset breast cancer
---

One of the major benefits of using Decision Trees is their interpretability. To take advantage of this benefit, you need to know how to pull out the information in a usable way.

The notebook where I did my code for this entry can be found on my github page in the [Entry 49 notebook](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/49a_nb_trees_analysis_interpt.ipynb).

## The Problem

One of the major challenges of Decision Trees is the propensity to overfit. To address this challenge, you need to know if there are any indicators of overfitting, when a tree may need to be pruned, or whether there are features with information that wouldn't be available when predicting on new data.

I'm also interested in ways to compare trees to each other.

I'll be working with classification trees for this post and the accompanying notebook. The same concepts should apply the regression trees, but I didn't prove that out with actual code.

## The Options

### Scoring Metrics

The obvious place to start is with the scoring metrics I discussed back in Entries [23](https://julielinx.github.io/blog/23_class_score_theory/) and [24](https://julielinx.github.io/blog/24_class_score_implement/). Using the [Entry 24 notebook](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/24_nb_class_score_implement.ipynb) I was able to quickly throw together a function that returned 10 scoring metrics for several cross validated versions of a model.

*Side note*: It's kinda fun to come back to old code and turn what had been a hard coded solution into a single function. I also combined the scoring metrics that are available with a couple of the ones I added using the `make_scorer` function.

So that gives me a way to rate a model and cross validate it. However, I also want to be able to look at some meta data type information about a single tree.

### Feature Importance

Feature importance is a value that gives you an idea of how important a specific feature is in the Decision Tree. All values are between 0 and 1. A value of 0 is no contribution and a value of 1 is perfect contribution. Values closer to 1 are less likely as all values add up to 1 which would make that feature a perfect way to predict. *Side note*, if the value is 1 then that feature probably has *data leakage*.

*Data leakage* is when information from the target variable is inappropriately captured in another feature. An example would be if the facility id in a cancer dataset perfectly predicted whether a patient lived or died within 5 years of diagnosis. This happened during a Data Science competition, the root cause of the facility id predicting longevity so well was that the facilities took different kinds of cancer or different stages of cancer. Some of those cancers or conditions were much more lethal than others, so while the correlation wasn't perfect, it was a very good predictor.

Page 180 of [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0) points out one of the practicalities of features with higher importance:

> Intuitively, predictors that appear higher in the tree (i.e., earlier splits) or those that appear multiple times in the tree will be more important than predictors that occur lower in the tree or not at all.

The `.feature_importance_` method returns the feature importance values for us. 

When initially looking at the results they seem to return the same information for trees that coefficients do for linear models. However, whereas coefficients in linear models are positive or negative (indicating if they contribute toward positively identifying or negatively disqualifying a feature), all feature importance values for trees are positive.

This means that feature importance doesn't tell you which class the feature is important for ("survived" vs "died") or how that feature is informative (high, low, etc). 

![Feature importance trees](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/49a_feature_importance.png?raw=true)

As discussed in [Entry 44](https://julielinx.github.io/blog/44_decision_trees/) Decision Trees automatically perform feature selection, meaning that not all features are used (i.e. they have an importance of 0). To make the results of `.feature_importance_` more readable in the notebook, I removed the features that weren't used, then listed the remaining features in order. As a final touch, I threw the values for the used features into a horizontal bar chart.

### Tree and Node Metrics

While working on [Entry 47](https://julielinx.github.io/blog/47_trees_pruning/), I looked at a series of metrics around the depth, number of leaves, and the impurity and sample size at the split and leaf levels. The `tree.tree_` method holds several of these full tree metrics including depth, number of nodes, number of leaves, number of classes, and number of features. It also includes several arrays of information on the impurity, sample size, and thresholds.

Combining this information with some edited code from Scikit Learn's [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#), I was able to grab the min, max, mean, median, and standard deviation of the basic metrics I'd looked at in [Entry 47](https://julielinx.github.io/blog/47_trees_pruning/).

Using these metrics I can see if the tree trained until all nodes were pure (all leaves have a Gini value of 0), how deep the tree goes, and what features are important in the Decision Tree.

## The Proposed Solution

Between the scoring metrics, feature importance information, and tree and node metrics I was able to compile a comprehensive view of a tree. These allow me to make informed decisions on whether the model has overfit, if it has become uninterpretable, and/or if it suffers from data sensitivity.

### Probability

I wanted to mention that the Decision Tree can return the probability that an observation belongs to a certain class during prediction. I didn't incorporate this into my notebook, but it's possible if you find it useful. The model does this by returning the ratio of training instances for the node the observation falls into.

For example, if an observation fell into the leaf node in the lower left of the tree trained on the Titanic data, the probability would be $\frac{\text{majority class}}{\text{number of samples}} = \frac{49}{54} \approx 0.91$

To do this in Scikit Learn, use the `predict_proba` method on the trained tree.

## Up Next

[Decision Tree subtypes](https://julielinx.github.io/blog/50_trees_subtypes/)

## Resources

- [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [Entry 23: Scoring Classification Models - Theory](https://julielinx.github.io/blog/23_class_score_theory/)
- [Entry 24: Scoring Classification Models - Implementation](https://julielinx.github.io/blog/24_class_score_implement/)
- [Entry 24 notebook - Scoring Classification Models - Implementation](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/24_nb_class_score_implement.ipynb)
- [Entry 44](https://julielinx.github.io/blog/44_decision_trees/)
- [Entry 47](https://julielinx.github.io/blog/47_trees_pruning/)
