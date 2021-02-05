---
title: "Entry 51: Ensemble Methods: Bagging, Boosting, Stacking"
categories:
  - Blog
tags:
  - ensembles
  - trees
  - supervised learning
---

There are multiple ways to reduce the bias and variance of Ensemble Learning, the three most common are bagging, boosting, and stacking. For more on bias and variance, see [Entry 21](https://julielinx.github.io/blog/21_reg_score_theory/).

## Bagging and Pasting

Bagging and pasting both involve training models using subsets of the training data, the only difference is whether sampling is done *with replacement* or *without replacement* (see [Entry 51](https://julielinx.github.io/blog/51_ensembles_models/) for definitions of replacement).

- Bagging (as seen below, this is also called *bootstrapping*): training data sample is chosen using *with replacement*
- Pasting: training data sample is chosen using *without replacement*

### Out-of-Bag

On page 195, [Hands-On Machine Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) has a nice section on the percentage of data chosen when using the bagging method and the concept of out-of-bag and out-of-bag evaluation:

> By default a `Bagging Classifier` samples *m* training instances with replacement (`bootstrap=True`), where *m* is the size of the training set. This means that only about 63% of the training instances are sampled on average for each predictor. The remaining 37% of the training instances that are not sampled are called *out-of-bag* (oob) instances. Note that they are not the same 37% for all predictors.

> Since a predictor never sees the oob instances during training, it can be evaluated on these instances, without the need for a separate validation set. You can evaluate the ensemble itself by averaging out the oob evaluations of each predictor.

> In Scikit-Learn, you can set `oob_score=True` when creating a `BaggingClassifier` to request an automatic oob evaluation after training.

*Further reading*: [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0) also discusses *out-of-bag* on page 197.

### Comparison

Which method to chose? *Hands-On Machine Learning* has a good breakdown comparing the two methods on page 195:

> Bootstrapping introduces a bit more diversity in the subsets that each predictor is trained on, so bagging ends up with a slightly higher bias than pasting; but the extra diversity also means that the predictors end up being less correlated, so the ensemble's variance is reduced. Overall, bagging often results in better models, with explains why it is generally preferred.

*Applied Predictive Modeling* puts it this way on pages 192 and 194:

> bagging effectively reduces the variance of a prediction through its aggregation process [...]. For models that produce an unstable prediction, like regression trees, aggregating over many versions of the training data actually reduces the variance in prediction and, hence, makes the prediction more stable.

However, it goes on and helps explain why non-Decision Tree algorithms aren't as effective when bagged:

> Bagging stable, lower variance models like linear regression and MARS, on the other hand, offers less improvement in predictive performance.

Since the major benefit is lower variance, if the underlying algorithms already have low variance, they won't experience as much improvement.

Bagged and pasted models are non-sequential ensemble learning techniques. This means that we can take advantage of parallel processing.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/ensemble_nonsequential.png?raw=true'>

## Boosting

Boosting (originally called *hypothesis boosting*) generally trains models sequentially, each new model attempting to correct the mistakes of its predecessor.

<img src='https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/ensemble_sequential.png?raw=true'>

From a scaling point of view, this severely hinders parallel processing, reducing scalability. Whereas bagged models can be trained on separate CPUs or even separate servers all at the same time, boosted models have to be trained one by one with the results of the previous model accounted for in the training data for the next model.

There are several ways to do this, but the two most popular are:

- Gradient Boosting
- AdaBoost (Adaptive Boosting)

I'll cover the two main algorithms in more detail in their own entries, but here is a quick and dirty explanation.

### Gradient Boosting

Gradient Boosting tries to fit a new model to the *residual errors* of the previous model (I talk about residuals in [Entry 21](https://julielinx.github.io/blog/21_reg_score_theory/)).

### AdaBoost

The idea behind AdaBoost is that it gives a higher weight to misclassified observations at each subsequential model training.

## Stacking

I gave a brief overview of stacking in the Algorithm Voting section of [Entry 51](https://julielinx.github.io/blog/51_ensembles_models/). As a reminder, in stacking you train a model using the predictions of the underlying ensemble models as the input.

So, if the predictions of the underlying ensemble models are "yes", "no", "yes", those would be the input to train another model. I imagine you could use cross validation to validate the accuracy of your results, but page 209 of *Hands-On Machine Learning* recommends using hold out sets. It also states that there's nothing stopping you from training multiple models on the output of the ensemble (essentially creating an ensemble of ensembles), but that starts to sound like Deep Learning to me.

## Up Next

Social Networks

## Resources

- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Entry 21: Scoring Regression Models - Theory](https://julielinx.github.io/blog/21_reg_score_theory/)
- [Entry 51: Ensemble Learning](https://julielinx.github.io/blog/51_ensembles_models/)
