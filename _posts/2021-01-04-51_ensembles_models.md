---
title: "Entry 51: Ensemble Learning"
categories:
  - Blog
tags:
  - ensembles
  - trees
  - supervised learning
  - machine learning
---

Ensemble techniques began to appear in the 1990s according to page 192 of [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0). Most people think of Random Forests (basically a bunch of Decision Trees that vote on the right answer) when they think of ensemble learning, but I wanted to discuss the concept of Ensemble Learning first, then go into different implementations.

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

Ensemble Learning works like the *wisdom of the crowd* phenomenon. *Wisdom of the crowd* is what happens when many people answer the same question, then those answers are aggregated into a final answer; like "Ask the Audience" from *Who Wants to be a Millionaire*. The thing about this strategy is that the aggregated answer is often better than, not just any one *person*, but better than the response of any single *expert*.

In Ensemble Learning, instead of getting responses from people, you get responses from models. The predictions from the various models are then aggregated into a final answer. Part of what makes this work is having a diverse set of models that are independent of each other. This is very similar to the statistical concept of Independent and Identically Distributed for input variables, which was discussed in [Entry 10](https://julielinx.github.io/blog/10_reorder_and_predict/).

[Hands-On Machine Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) emphasizes this point on page 191. There are various ways this can be achieved, which include:

- Use diverse algorithms
- Train models using different subsets of the data
  - Subsets of observations
    - Without replacement
    - With replacement
  - Subsets of features

### Diverse Algorithms

Algorithms can include pretty much anything: decision trees, linear or logistic regression, SVMs, etc. When all of the involved models are Decision Trees, this is called Random Forest (get it? A bunch of trees makes a forest). Despite the fact that a Random Forest model contains all Decision Trees, *Hands-On Machine Learning* points out on page 189 that it is one of the most powerful algorithms currently available.

[Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413) states on page 85 that are are only two types of ensemble models that have proven effective on a wide range of datasets:

- Random Forest
- Gradient Boosted Decision Trees

This is 100% not going to stop me from trying different variations for myself.

*Fun fact*, the Random Forest algorithm, while an ensemble method in its own right, can also be included as part of a larger ensemble that includes other algorithm types.

### Subsets of Data

#### Replacement

When training on different subsets, there are two ways to sample the data:

  - Without replacement
  - With replacement
  
*Replacement* basically indicates whether each observation can be included more than once in a data sample.

A super simple example of bagging (I'll discuss bagging in [Entry 52](https://julielinx.github.io/blog/52_ensembles_bag_boost_stack/), for now just think of it as sampling with replacement) from page 86 of *Introduction to Machine Learning with Python*:

> To illustrate, let's say we want to create a bootstrap sample of the list ['a', 'b', 'c', 'd']. A possible bootstrap sample would be ['b', 'd', 'd', 'c']. Another possible sample would be ['d', 'a', 'd', 'a'].

Let's look at both with and without replacement using a slightly larger dataset. The example I like best uses a bag of marbles.

Let's say we have a bag of 5 green, 5 blue, and 5 yellow marbles (each marble representing an observation in our training data). The two replacement methods work as follows:

**Without replacement**

When a marble is removed from the bag, it is noted as part of the sample then left out of the bag, making it unavailable to be chosen again. Here are two variations we could have for samples to train three models:
- Variation one:
  - Sample 1: 3 blue, 3 green, 4 yellow
  - Sample 2: 3 blue, 4 green, 3 yellow
  - Sample 3: 4 blue, 3 green, 3 yellow
- Variation two:
  - Sample 1: 4 blue, 1 green, 5 yellow
  - Sample 2: 2 blue, 5 green, 3 yellow
  - Sample 3: 5 blue, 5 green, 0 yellow
- If you care to add up marbles, you'll find:
  - Each sample in the variation has 10 marbles (I used 2/3rds of the marbles per training subset)
  - Each color has a maximum of 5 per sample (because there are only 5 marbles of each color) and a minimum of 0

**With replacement**

When a marble is removed from the bag, it is noted as part of the sample then returned to the bag, making it available to be chosen again. Here are two variations we could have for samples to train three models:
- Variation one:
  - Sample 1: 7 blue, 7 green, 6 yellow
  - Sample 2: 7 blue, 6 green, 7 yellow
  - Sample 3: 6 blue, 7 green, 7 yellow
- Variation two:
  - Sample 1: 15 blue, 5 green, 0 yellow
  - Sample 2: 0 blue, 20 green, 0 yellow
  - Sample 3: 5 blue, 8 green, 7 yellow
- If you care to add up the marbles, you'll find:
  - Each group in the variation has 20 marbles (because I replaced each marble, I wasn't limited to 15 marbles)
  - The maximum number of times any one marble can be chosen is the size of the training set (like the group with 20 green marbels and no other color) and the minimum number of times any one marble can be chosen is 0
    - Keep these minimums and maximums in mind, because it means some observations may not be used at all, while others may be used more than once
    
#### Random Subspaces and Random Patches

In addition to sampling the observations, we can also sample the features. Unlike the individual observations, there is no with or without replacement. When sampling features, it's a straight subset of the full feature set. If we were to use a feature more than once (as is possible when using with replacement) we'd run into problems with collinearity, as explained in [Entry 7](https://julielinx.github.io/blog/07_collinearity/).

Each model is trained on a different random subset of the features, providing additional predictor diversity. *Hands-On Machine Learning* talks about how feature sampling can be used by itself (random subspaces) or in conjunction with observation sampling (random patches) on page 196.

**Random Subspaces**

Uses all observations in the training dataset and only samples the features. In the below dataset, the columns highlighted in blue would be used to train a model.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/random_subspaces.png?raw=true'>

**Random Patches**

Uses sampling of both the observations and the features. In the below dataset, the observation/feature combinations highlighted in green would be used to train a model.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/random_patches.png?raw=true'>

*Side note*, in case you were interested, the data in the above spreadsheets is from the UCI Repository's [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom).

## Purpose

On page 426 [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) points out that Decision Trees tend to overfit the data. These less generalizable individual models can be combined to counteract the effect of the overfitting and result in an ensemble that predicts well on previously unseen data.

*Hands-On Machine Learning* puts it this way on page 192:

> Once all predictors are trained, the ensemble can make a prediction for a new instance by simply aggregating the predictions of all predictors. The aggretaion function is typically the *statistical model* (i.e., the most frequent prediction, just like a hard voting classifier) for classification, or the average for regression. Each individual predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance. Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the original training set.

This is nicely illustrated in Aurelien's example from the [Hands-On Machine Learning Chapter 7 GitHub page](https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb):

<img src='https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/bagging_vs_not.png?raw=true'>

## Behavior

The "votes" (i.e., predictions) of the models can be counted, or aggregated, in three different ways: *hard voting*, *soft voting*, or *stacking*. 

### Algorithm Voting

Once the ensemble is formed, there are three ways to aggregate the predictions:

- Hard voting
- Soft voting
- Stacking

**Hard voting**: takes the answer that has the most votes. So if 18 models vote "no" and 21 vote "yes" the final answer would be "yes".

**Soft voting**: takes the average prediction probability and uses that to return the final answer, which allows you to take into account the confidence of the underlying models. For example, if 5 models had the following prediction probabilities: 0.8, 0.4, 0.9, 0.1, 0.7, then you'd average the votes $\frac{0.8 + 0.4 + 0.9 + 0.1 + 0.7}{5} = 0.58$. As you can see, while most models were voting for the upper range (0.8, 0.9, 0.7), the extreme confidence of one of the lower votes (0.1, 0.4) skewed the model lower, but still with an overall vote toward the higher range.

This becomes more important when there are many low confidence votes and few high confidence votes. An example would be 6 models with the following prediction probabilities: 0.55, 0.48, 0.59, 0.41, 0.50, 0.95, then you'd average the votes $\frac{0.52 + 0.48 + 0.59 + 0.41 + 0.50 + 0.95}{6} = 0.575$ Notice the first five predictions average out to 0.50, it's the last, most confident model that is the decisive vote.

Soft voting is only available when all the underlying algorithms return the prediction probability. In Scikit Learn this means that the algorithm has to have a `predict_proba` method or some way to set a `probability` hyperparameter (like SVC). Then the `voting` parameter can be set to `soft` in the ensemble classifier.

**Stacking**: short for *stacked generalization* and is also called *blending* or *meta learning*. The general principle is that instead of using a simple method like soft or hard voting, you train a model using the predictions of the underlying ensemble models as the input.

I'll cover stacking in a little more detail in the next post about three commonly used ensemble methods:

- Bagging/pasting
- Boosting
- Stacking

### Weak and Strong Learners

Ensemble models can be built and aggregated weather the underlying models are *weak learners* or *strong learners*.

A *weak learner* is a model that performs only slightly better than random guessing. For unbalanced datasets where the target class of interest is the minority class, this can mean that the model performs worse than 50/50. Take the dataset I generally work with, the target class I'm interested in generally only makes up 4-8% of my data. A weak learner would be a model that could make correct predictions as little as 5-9% of the time.

When there are a sufficient number of diversely trained models, the ensemble model can actually perform quite well, making the ensemble a *strong learner* (a model that performs well on its own).

*Hands-On Machine Learning* puts it this way on page 181:

> suppose you build an ensemble containing 1,000 classifiers that are individually correct only 51% of the time (barely better than random guessing). If you predict the majority voted class, you can hope for up to 75% accuracy! However, this is only true if all classifiers are perfectly independent, making uncorrelated errors, which is clearly not the case because they are trained on the same data. They are likely to make the same types of errors, so there will be many maority votes for the wrong class

So, while this method is powerful, there are still challenges to watch out for.

### Parallel Processing

On page 193 *Hands-On Machine Learning* points out a very big advantage of non-sequential ensemble learning (i.e., the training of one model is independent of the other models): Because models that are part of an ensemble don't rely on the outcome of their fellow models, the models "can all be trained in parallel, via different CPU cores or even different servers". This means that ensemble models scale well.

## Up Next

[Ensemble Methods: Bagging, Boosting, Stacking](https://julielinx.github.io/blog/52_ensembles_bag_boost_stack/)

## Resources

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Hands-On Machine Learning Chapter 7 GitHub page](https://github.com/ageron/handson-ml2/blob/master/07_ensemble_learning_and_random_forests.ipynb)
- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Entry 10: Reorder Pre-processing and Make Predictions](https://julielinx.github.io/blog/10_reorder_and_predict/)
- [Entry 7: Collinearity](https://julielinx.github.io/blog/07_collinearity/)
- [Entry 52: Ensemble Methods: Bagging, Boosting, Stacking](https://julielinx.github.io/blog/52_ensembles_bag_boost_stack/)
- [UCI Repository's Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)
