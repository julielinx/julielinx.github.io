---
title: "Entry 50: Decision Tree Subtypes"
categories:
  - Blog
tags:
  - trees
  - supervised learning
---

[The Complete Guide to Decision Trees](https://towardsdatascience.com/the-complete-guide-to-decision-trees-28a4e3c7be14) sums up the similarities between Decision Tree Algorithms very succinctly:

> All [Decision Trees] perform basically the same task: they examine all the attributes of the dataset to find the ones that give the best possible result by splitting the data into subgroups. They perform this task recursively by splitting subgroups into smaller and smaller units until the Tree is finished (stopped by certain criteria).

But there are multiple ways to achieve these tasks. The variations cover several aspects ranging from the number of splits (binary or multiway splits) to pruning strategies to impurity metrics.

## The Problem

My books don't have much to say about the different algorithms. Scikit Learn uses the CART algorithm exclusively, so most of the information in them is tailored to that algorithm. But I keep seeing other algorithms referenced, so I want to have an idea about how they differ and what their strengths and weaknesses are.

## Decision Tree Algorithm Quick Reference

Based on the quote from *The Complete Guide to Decision Trees*, I expect all of the Decision Tree implementations to be *greedy*, which I discussed in [Entry 44](https://julielinx.github.io/blog/44_decision_trees/), citing page 180 of [Hands-On Machine Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646) (which was specifically talking about the CART algorithm):

> [the algorithm] searches for an optimum split at the top level, then repeats the process at each subsequent level. It does not check whether or not the split will lead to the lowest possibly impurity several levels down. A greedy algorithm often produces a solution that’s reasonably good but not guaranteed to be optimal.

The Decision Tree algorithms I'll summarize are five of the most prominent algorithms. Here's a quick reference table summarizing their similarities and differences:

<table align='left'>
    <tr>
        <td><b>Algorithm</b></td>        
        <td><b>Full Name</b></td>        
        <td><b>Year Developed</b></td>
        <td><b>Impurity Measure</b></td>
        <td><b>Prediction Types</b></td>        
        <td><b>Feature Use</b></td>
        <td><b>Handles Missing Values</b></td>
        <td><b>Value Handling</b></td>
        <td><b>Pruning</b></td>
        <td><b>Split Handling</b></td>
    </tr>
    <tr>
        <td><b>CHAID</b></td>
        <td>Chi-square Automatic Interaction Detection</td>
        <td>1980</td>
        <td>Chi squared test (classification) or F-test (continuous data)</td>
        <td>Classification and Regression</td>
        <td></td>
        <td>Yes</td>
        <td></td>
        <td>None</td>
        <td>Multiway splits</td>
    </tr>
    <tr>
        <td><b>CART</b></td>
        <td>Classification and Regression Trees</td>
        <td>1984</td>
        <td>Gini Index (classification) or Least Squares Deviation (continuous values)</td>
        <td>Classification and Regression</td>
        <td>Can use the same feature multiple times</td>
        <td>Yes</td>
        <td>Handles continuous and categorical values</td>
        <td>Pre-pruning</td>
        <td>Binary splits</td>
    </tr>
    <tr>
        <td><b>ID3</b></td>
        <td>Iterative Dichotomiser</td>
        <td>1986</td>
        <td>Entropy / Information Gain</td>
        <td>Mainly classification, struggles with regression</td>
        <td>Uses each feature once or less</td>
        <td>No</td>
        <td>Can't handle continuous data</td>
        <td></td>
        <td>Multiway splits</td>
    </tr>
    <tr>
        <td><b>C4.5 / C5.0</b></td>
        <td>C4.5 / See5</td>
        <td>1993</td>
        <td>Gain Ratio</td>
        <td>Classification and Regression</td>
        <td></td>
        <td>Yes</td>
        <td>Handles continuous and categorical values</td>
        <td>Post-pruning</td>
        <td>Multiway splits</td>
    </tr>
    <tr>
        <td><b>MARS</b></td>
        <td>Multivariate Adaptive Regression Splines</td>
        <td>1991</td>
        <td>Linear regression with hinge functions</td>
        <td>Mainly regression, but can adapt to classification with logistic regression</td>
        <td></td>
        <td>Sometimes: depends on programmatic implementation</td>
        <td>Handles continuous and categorical values</td>
        <td>Yes</td>
        <td>Hinge functions</td>
    </tr>
</table>

# The Algorithms

## CHAID - Chi-square Automatic Interaction Detection

According to the Seminar Paper available in Michael Dorner's [Decision Trees repo](https://github.com/michaeldorner/DecisionTrees), CHAID was proposed in 1979 by Gordon Kass as a modification to the Automatic Interaction Detection (AID) algorithm developed by John Sonquist and James Morgan in 1964. This makes the the oldest (and certainly based on the oldest) algorithm that I'll discuss. *The Complete Guide to Decision Trees* states that CHAID is one of the oldest Decision Tree algorithms that produces multiway splits and is suitable for classification and regression tasks.

### Description

According to [Decision Trees Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html) CHAID finds the statistical significance of the differences between child and parent nodes. These differences are measured using Chi-squared independence tests for classification tasks and the F-test for regression tasks. The higher the value, the more statistically significant the difference is between the child and parent nodes.

### Purpose

To generate a classification tree with the most statistically significant splits.

### Behavior

*The Complete Guide to Decision Trees* states that features that aren't significantly different from each other (with respect to the target variable) are merged together.

When handling missing values CHAID doesn't replace them, it treats them as a single class and will use the same merging strategy as it does for other features.

### Strengths

- Handles missing values
- Easy to manage

### Weaknesses

- Tends to produce wide trees (as opposed to deep trees), which may not translate well to real world conditions
- No pruning capability
- Not as powerful at detecting small difference as other algorithms
- Not as fast as other algorithms

## CART - Classification and Regression Trees

### Description

Decision Tree algorithm that creates binary trees using the Gini Index as a the impurity measure for classification and Least Square Deviation for regression.

### Purpose

CART seems to have been developed to address deficiencies in the algorithms of the time. Benefits listed in *The Complete Guide to Decision Trees* and [1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart) include:

- Handling missing data
- Handling both numeric and categorical variables
- Predicting on both classification and regression tasks
- Handling raw data (no preprocessing required)
- Using the same features multiple times to uncover complex interdependencies.

*1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART* points out that the optimized version used by Scikit Learn doesn't do all of these things. For example, it can't currently handle categorical features.

### Behavior

The Scikit Learn implementation of CART allows the user to specify either Gini Index or Entropy as the impurity measure. ID3 (the next algorithm discussed) also uses Information Gain (which is directly related to Entropy). Since these two measures are so prominent, I thought it'd be nice to have a quick way to compare them. Fortunately, [Decision Tree Flavors: Gini Index and Information Gain](http://www.learnbymarketing.com/481/decision-tree-flavors-gini-info-gain/#:~:text=Summary%3A%20The%20Gini%20Index%20is,of%20each%20class%20from%20one.&text=Information%20Gain%20multiplies%20the%20probability,partitions%20with%20many%20distinct%20values.) has nice, bullet pointed lists of how Gini Index and Information Gain/Entropy behave. I put these into a table for easy comparison.

<table align='left'>
    <tr>
        <td><b>Gini Index</b></td>
        <td><b>Information Gain / Entropy</b></td>        
    </tr>
    <tr>
        <td>Favors larger partitions</td>
        <td>Favors splits with small counts but many unique values</td>
    </tr>
    <tr>
        <td>Uses squared proportion of classes</td>
        <td>Weights probability of class by log(base=2) of the class probability</td> 
    </tr>
    <tr>
        <td>Perfectly classified, Gini Index would be zero</td>
        <td>Smaller values of Entropy are better</td>
    </tr>
    <tr>
        <td>Results are capped at 1 (Range of values is 0 to 1)</td>
        <td>The maximum value for Entropy depends on the number of classes. It's based on base-2 (2 classes: max = 1, 4 classes: max = 2, 8 classes: max = 3, 16 classes: max = 4)</td>
    </tr>
    <tr>
        <td></td>
        <td>Information Gain is the Entropy of the parent node minus the entropy of the child nodes</td>
    </tr>
</table>

### Strengths

- Supports continuous numeric features
- Capable of Regression or Classification tasks
- Handles missing data
- Minimal preprocessing required
- Allows a feature to be used multiple times for split decisions

### Weaknesses

- The Scikit implementation doesn't support categorical variables
- Allows a feature to be used multiple times for split decisions

## ID3 - Iterative Dichotomiser

Ross Quinlan developed the ID3 algorithm in 1986 as an improvement of older Decision Tree algorithms.

### Description

According to [thefreedictionary.com](https://www.thefreedictionary.com/dichotomisation#:~:text=Noun,act%20or%20process%20of%20dividing), "dichotomisation" is "the act of dividing into two sharply different categories."

As such, it makes sense that one of the early Decision Tree algorithms used this term in its name. However, since ID3 uses multiway splits and can have more than two children nodes per parent node, the name is a bit of a "gotcha" unless you ignore the "two" part of the definition.

### Purpose

*1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART* has the following information about ID3:

> The algorithm creates a multiway tree, finding for each node (i.e. in a greedy manner) the categorical feature that will yield the largest information gain for categorical targets. Trees are grown to their maximum size and then a pruning step is usually applied to improve the ability of the tree to generalise to unseen data.

*Decision Trees Explained* lists out the steps of the algorithm:

1. It begins with the original set S as the root node
2. On each iteration of the algorithm, it iterates through every unused attribute of the set S and calculates Entropy(H) and Information gain(IG) of this attribute
3. It then selects the attribute which has the smallest Entropy or Largest Information gain
4. The set S is then split by the selected attribute to produce a subset of the data
5. The algorithm continues to recur on each subset, considering only attributes never selected before

The tendency to state that ID3 uses both Entropy and Information Gain as the impurity measures can be confusing. *Decision Trees Explained* eloquently resolves this confusion by explicitly stating that

> Information gain is a decrease in entropy. It computes the difference between entropy before split and average entropy after split of the dataset based on given attribute values.

As such, whether the programmatic implementation uses Entropy or Information Gain, it's looking at the same basic information. The major difference is whether the algorithm is optimized to find the lowest Entropy or the highest Information Gain.

### Behavior

Based on the information in the Purpose section, we can call out several behaviors of this algorithm:

- It's a greedy algorithm, making the best choice at any particular point without validating later in the tree or backtracking
- The impurity measures are Entropy and Information Gain
- It uses multiway splits instead of just binary splints
- When pruning is applied, it's generally as a post-pruning process
- It's used on categorical data

### Strengths

- Handles multiway splits
- Uses each feature only once to make split decisions
- Since each feature is only used once, the tree is usually relatively small

### Weaknesses

- Doesn't perform well with continuous / numeric values
  - Mainly used for classification tasks
  - *The Complete Guide to Decision Trees* points out that techniques like binning numeric values can improve performance on Regression Trees
- Can't handle missing values
- Information Gain is biased toward features with a large number of values
- Uses each feature only once to make split decisions

## C4.5 and C5.0

Ross Quinlan developed the C4.5 algorithm in 1993 as an improvement to his ID3 algorithm. 

### Description

C4.5 is the improved successor of the ID3 algorithm. The major improvements are that the algorithm bins numeric variables into discrete attributes, thus removing the limitation around continuous numeric vs discrete categorical variables, and the use of Gain Ratio instead of Information Gain.

C5.0 is the commercial successor to C4.5. It has several improvements around optimized runtime, memory usage, and reduced complexity. There's a really nice comparison of the results of the two algorithms at [Is See5/C5.0 Better Than C4.5?](https://rulequest.com/see5-comparison.html).

### Purpose

According to *1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART*, C4.5 turns a trained tree into a series of if-then rules. These rules are then evaluated against each other to determine the order, using the most accurate rules first.

*Decision Trees Explained* makes the observation that Information Gain is biased towards attributes with a large number of values when choosing the root node, so it tends to choose features with a large number of distinct values. Gain Ratio helps overcome this problem by taking into account the number of branches that would result. "It corrects information gain by taking the intrinsic information of a split into account."

### Behavior

- Pruning is done during the evaluation of the rules. *1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART* states that a rule’s precondition is removed if the accuracy of the rule improves without it.
- Utilizes "windowing" (from *The Complete Guide to Decision Trees*)
  - Windowing was originally developed to overcome memory limitations of early computers
  - Similar to cross validation in that a subset of data is used
  - Misclassified data points are added to the subset of training data and the decision tree is retrained
  - This retraining is repeated until all data points are correctly classified
  - The process allows the algorithm to capture "rare" instances in addition to a random set of "ordinary" instances
- Uses error rate to prune the tree (per *The Complete Guide to Decision Trees*)

### Strengths

- Can handle continuous numeric data, making it suitable for both Classification and Regression Trees
- Improved feature impurity measure: Gain Ratio
- Handles missing data
- Capable of pruning
- Includes a way of addressing "rare" cases

### Weaknesses

- Overfitting

## MARS - Multivariate Adaptive Regression Splines

MARS was developed by Jerome Friedman in 1991.

### Description

MARS is a form of regression analysis that fits piecewise linear regressions using hinge functions. Based on the examples on the [Multivariate adaptive regression splines](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline) Wikipedia page, it looks like this basically creates linear models on subsets of the data, using hinge functions to pivot from one linear regression to another.

![Linear vs MARS](https://github.com/julielinx/datascience_diaries/blob/master/03_supervised_learning/02_tree_based/images/mars_wikipedia_ex.png?raw=true)

### Purpose

This algorithm takes into account nonlinear relationships while benefiting from the strengths of linear models.

### Behavior

According to [Wikipedia](https://en.wikipedia.org/wiki/Predictive_analytics#Multivariate_adaptive_regression_splines), MARS deliberately overfits the model and then prunes to get to the optimal model.

### Strengths

The *Multivariate adaptive regression splines* Wikipedia page includes a nice list of strengths and weaknesses.

- More flexible than linear regression
- Easy to understand and interpret
- Handles continuous and categorical data
- Better than recursive partitioning (like CART models) because hinges are more appropriate for numeric variables than the piecewise constant segmentation used by recursive partitioning
- Minimal data preprocessing
- Effect of outliers is diminished
- Automatic feature selection (uses important features, excludes unimportant ones)

### Weaknesses

- Computational intensive
- Automatic feature selection can be arbitrary, especially with correlated features
- Correlated features can make interpretability difficult
- Much slower than recursive partitioning (like CART)
- Missing value handling depends on the implementation (`earth`, `mda`, and `polspline` can't accept missing values, but `rpart` and `party` can)

## The Fail

This entry got a little sloppy. After I started researching the different algorithms, my interest waned since I couldn't run them myself. Without a way to run them I couldn't experiment with parameters nor could I see how the results varied between different implementations. Thus I couldn't form an opinion of one algorithm vs another for myself.

I almost abandoned the post, but I didn't want to lose what I'd already found out. So, take what you can from this entry. If you're interested in learning more, check out the resource links.

## Up Next

Ensemble Models

## Resources

- [The Complete Guide to Decision Trees](https://towardsdatascience.com/the-complete-guide-to-decision-trees-28a4e3c7be14)
- [Decision Trees Explained](https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html)
- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Decision Trees - An Introduction](https://github.com/michaeldorner/DecisionTrees)
- [ID3 algorithm](https://en.wikipedia.org/wiki/ID3_algorithm)
- [C4.5 algorithm](https://en.wikipedia.org/wiki/C4.5_algorithm)
- [Predictive analytics](https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29)
- [1.10.6. Tree algorithms: ID3, C4.5, C5.0 and CART](https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart)
- [Is See5/C5.0 Better Than C4.5?](https://rulequest.com/see5-comparison.html)
- [A Step by Step ID3 Decision Tree Example](https://sefiks.com/2017/11/20/a-step-by-step-id3-decision-tree-example/)
- [decision-tree](https://github.com/ryanmadden/decision-tree)
- [Decision Tree Flavors: Gini Index and Information Gain](http://www.learnbymarketing.com/481/decision-tree-flavors-gini-info-gain/#:~:text=Summary%3A%20The%20Gini%20Index%20is,of%20each%20class%20from%20one.&text=Information%20Gain%20multiplies%20the%20probability,partitions%20with%20many%20distinct%20values.)
- [Multivariate adaptive regression splines](https://en.wikipedia.org/wiki/Multivariate_adaptive_regression_spline)
