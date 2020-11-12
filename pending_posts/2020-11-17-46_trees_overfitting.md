---
title: "Entry 46: Overfitting, Underfitting, and Data Sensitivity"
categories:
  - Blog
tags:
  - trees
  - supervised learning
---

I wanted to point a co-worker to information about overfitting the other day. While I've discussed it in entries [17](https://julielinx.github.io/blog/17_resampling/) and [30](https://julielinx.github.io/blog/30_learning_curves_imp_perform/), I realized I haven't covered it in its own entry. Since overfitting is such a big problem with Decision Trees, this feels like the perfect time to stop and remedy that oversight.

## Overfitting and Underfitting 

Decision trees are very prone to overfitting. As a reminder from [Entry 17](https://julielinx.github.io/blog/17_resampling/) here are the definitions of overfitting and underfitting from [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413/ref=sr_1_15?keywords=scikit+learn&qid=1583195970&s=books&sr=1-15) by Andreas Muller and Sarah Guido.

- **Overfitting**: when a model is fit too closely to the training set, but is not able to generalize to new data, it is *overfit*.
- **Underfitting**: when a model is fit too loosely / simply, causing it to predict poorly on training and test data, it is *underfit*.

Aurelien Geron has a great illustration of this on page 184 of [Hands-On Machine Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) using a tree-based model on a regression dataset. The code for this chart can be found in Geron's GitHub [notebook for Chapter 6](https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb).

<img src='images/ml2_overfitting.png'>

The tree represented on the left is clearly overfit and wouldn't generalize well to data it hasn't seen before. The tree on the right will generalize to unseen data much better.

Jake VanderPlas has a great example of the propensity of classification tree-based models to overfit in his book [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) on page 424. Using a synthetic dataset created using the `make_blots()` function, he trained two models using different subsets of half the data. His full explaination and the code can be found in his accompanying [Github notebook](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb).

<img src='images/DSHandbook_tree_overfit.png'>

As you can see, there are areas where the models have the same results; the top and two lower regions. But the areas where the different groups overlap provide very different results.

This overfitting also illustrates the sensitivity of tree-based models to small changes in the data.

## Data sensitivity

Another challenge when using tree-based models is that they like decision boundaries that run up and down or side to side, not diagonal. This makes them sensitive to data rotation.

*Hands-On Machine Learning* has a great illustration of this on page 185, with the accompanying code in his [Chapter 6 notebook](https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb) on GitHub.

<img src='images/ml2_data_sensitivity.png'>

While both models split the data with the same accuracy, the one on the right will most likely have trouble generalizing to new data. Aurelien Geron recommends using Principal Component Analysis (PCA) to help overcome this challenge. He covers PCA in chapter 8, I'll get to it eventually in this blog.

## Up Next

Pruning Trees

## Resources

- [Introduction to Machine Learning with Python](https://www.amazon.com/Introduction-Machine-Learning-Python-Scientists/dp/1449369413)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Hands-On Machine Learning notebook: Chapter 6 – Decision Trees](https://github.com/ageron/handson-ml2/blob/master/06_decision_trees.ipynb)
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [Python Data Science Handbook notebook, chapter 5: In-Depth: Decision Trees and Random Forests](https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.08-Random-Forests.ipynb)
- [Entry 17: Resampling](https://julielinx.github.io/blog/17_resampling/)
- [Entry 30: Improve Performance - Learning Curves](https://julielinx.github.io/blog/30_learning_curves_imp_perform/)
