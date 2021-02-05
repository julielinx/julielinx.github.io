---
title: "Entry 48: Decision Tree Impurity Measures"
categories:
  - Blog
tags:
  - trees
  - supervised learning
  - n-grams
  - machine learning
---

Impurity seems like it should be a simple calculation. However, depending on prevalence of classes and quirks in the data, it's usually not as straight forward as it sounds.

## The Problem

To demonstrate the challenges in separating classes, let's pretend we're trying to identify twitter posts that were written by a bot. So we're trying to split the twitter posts into "bot" and "human" classes.
 
During an analysis of the data, we noticed that bots tend to use specific words or groups of words that humans don't. As such, we want to use n-grams to pull out these individual and grouped words.

### Example

#### Context

To create an *n-gram* you break a sentence, paragraph, or text document into manageable chunks where punctuation, common words, and capitalization have been removed.

Let's start with the following sentence:

> "The quick, brown fox jumped over the fence."

First, we strip out "the", replace any uppercase letters with lowercase, and get rid of the punctuation to end up with:

> "quick brown fox jumped over fence"

Next we break it into chucks, which gives us a list, the length of which depends on how many words are included in a chunk:

  - *unigram*: Treating each word as its own chunk
    - Example: ["quick", "brown", "fox", "jumped", "over", "fence"]
  - *bigram*: Grouping sets of two words together as a chunk
    - Example: ["quick brown", "brown fox", "fox jumped", "jumped over", "over fence"]
  - *trigram*: Grouping sets of two words together as a chunk
    - Example: ["quick brown fox", "brown fox jumped", "fox jumped over", "jumped over fence"]
  - *n-gram*: This chunking/grouping can be done with any number of words: 4, 5, 6, or more. As such, the generic term referring to any given number of chunks is *n-gram*

#### Results

Let's say we want to identify the unigrams that are most likely written by a bot, but only want to record the single *most* bot-like word for any individual tweet. Our pretend analysis returned three good candidates for the tweet "Submit your request today to terminate your cable bill and abort unwanted charges!":

- "submit"
  - 45 total uses
    - 30 bot uses
    - 15 human uses
  - 66.66% bot usage
- "terminate"
  - 5 total uses
    - 5 bot uses
    - 0 human uses
  - 100% bot usage
- "abort"
  - 35 total uses
    - 29 bot uses
    - 6 human uses
  - 82.86% bot usage

If we choose to use the metric of the word that is most used by bots we end up with "submit", but that word has the worst ratio of bot to human usage.

If we pick the word with the highest percentage of bot usage we end up with "terminate", which is used least frequently in our overall dataset and is less likely to be useful.

What we really want for our metric is something that balances the percentage with the frequency of use and would return "abort" as our most bot-like result.

## The Options

There are several different impurity measures for each type of decision tree:

### `DecisionTreeClassifier`

- **Default**: gini impurity
  - From page 234 of [Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)
    - $G(t) = 1 - \displaystyle\sum_{i=1}^{c} P_{i}^2$
    - Where
      - $G(t)$: gini impurity at node $t$
      - $t$: a specific node
      - $c$: class
      - $P_{i}$: proportion of observations of class $c$ at node $t$
  - From page 177 of [Hands-On Machine Learning](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
    - $G_{i} = 1 - \displaystyle\sum_{k=1}^{n} P_{i,k}^2$
    - Where
      - $G_{i}$: gini impurity at node $i$
      - $i$: a specific node
      - $k$: class
      - $P_{i, k}^2$: the raio of class $k$ instances among the training instances of node $i$
   - The trick to understanding gini impurity is to realize that the calculation is done with the numbers in `samples` and `values`
     - Example: Take the green setosa class node at depth 2
       - Samples = 44
       - Values = [0, 39, 5]
       - Gini = $1 - \frac{0}{44}^2 - \frac{39}{44}^2 - \frac{5}{44}^2 \approx 0.201$
   - Reading Gini impurity
     - A Gini impurity of 0 means that the node is pure
       - Example: If all the samples in the green setosa class node at depth 2 was in fact setosa we'd get: $1 - \frac{44}{44} = 1 - 1 = 0$
     - The closer the Gini impurity is to 1 the more impure (i.e. mixed) it is.
       - Example: If the classes in the green setosa class node at depth 2 were in fact evenly split we'd get: $1 - \frac{15}{45} - \frac{15}{45} - \frac{15}{45} \approx 0.67$
- **Alternate**: entropy
  - Per page 180 of *Hands-On Machine Learning*: "originated in thermodynamics as a measure of molecular disorder; entropy approaches zero when molecules are sill and well ordered."
  - A value of 0 means all samples belong to the same class (i.e. node is pure)
  - Higher values mean there is more mixing of the classes (i.e. node has higher impurity)
  - Similar to Gini Impurity, but includes a log component instead of a squared component
  - From page 181 of *Hands-On Machine Learning*
    - $H_{i} = - \displaystyle\sum_{\substack{k=1\\ P_{i, k} \neq 0}}^{n} P_{i, k} log_{2}(P_{i, k})$

### `DecisionTreeRegressor`

- **Default**: MSE (mean squared error)
- **Alternates**
  - Friedman MSE (MSE plus Friedman’s improvement score for potential splits)
  - MAE (mean absolute error)

## The Proposed Solution

The solution part is easy, default values have already been provided for both the Classifier and Regressor versions of Decision Trees in Scikit Learn.

*Hands-On Machine Learning* provides some context on the differences for Classification Trees on page 181: 
  - Using Gini or Entropy usually leads to similar trees
  - Gini is slightly faster to compute
  - When results are different
    - Gini impurity tends to isolate the most frequent class in its own branch
    - Entropy produces slightly more balanced trees

For nuanced comparisons between the different regression metrics, check out Entries [21](https://julielinx.github.io/blog/21_reg_score_theory/) and [22](https://julielinx.github.io/blog/22_reg_score_implement/) which both talk about scoring regression models, which includes MSE and MAE.

## Up Next

[Analyzing Trees](https://julielinx.github.io/blog/49_trees_analysis_interpt/)

## Resources

- [Machine Learning with Python Cookbook](https://www.amazon.com/Machine-Learning-Python-Cookbook-Preprocessing/dp/1491989386)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646)
- [Entropy: How Decision Trees Make Decisions](https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8)
- [Decision Tree Classification in Python](https://www.datacamp.com/community/tutorials/decision-tree-classification-python)
- [1.10.7. Mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)
- [Entry 21: Scoring Regression Models - Theory](https://julielinx.github.io/blog/21_reg_score_theory/)
- [Entry 22: Scoring Regression models - Implementation](https://julielinx.github.io/blog/22_reg_score_implement/)
