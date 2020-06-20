---
title: "Entry 28: Thresholds - Profit and cost"
categories:
  - Blog
tags:
  - model-eval
  - dataset titanic
---



## The Problem



## The Options

#### Profit

This was thrown into *[Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)* as an example of a non-accuracy based criteria.  It allocates a gain from TP and costs for FP and FN to assign a dollar amount. The example in the book was for a direct mail campaign. `x` amount was expected to be gained by customers that responded to the mailer, `y` was spent on each mailer, and `z` was the amount lost for mailers not sent to customers that would have responded.

The same basic equation can be used from a savings perspective in use cases such as fraud. `x` would be the expected savings for each case of fraud successfully identified and stopped, `y` would be costs like customers lost due to increased inconvenience, and `z` would be the amount lost for each fraudulent case gone undetected.

$profit = xTP - yFP - zFN$

An equation like this has potential for helping to set a threshold for the prediction.

Variations on this concept are the probability cost function and normalized expected cost.

#### Probability cost function (PCF)

*Applied Predictive Modeling* page 262:

> The PCF is the proportion of the total costs associated with a false-positive sample.

$PCF = \frac{P \times C(fn)}{P \times C(fp) + (1 - P) \times C(fn)}$

Where:

- *P* is the (prior) probability of the event (all positives)
  + IE: *P* is the proportion of positives in the data
  + As such, 1 - *P* is the probability of a non-event, or the proportion of all negatives in the data
- *C(fn)* is the cost of a false negative (positive observation predicted as a negative)
- *C(fp)* is the cost of a false positive (negative observation predicted as a positive)

#### Normalized expected cost (NEC)

*Applied Predictive Modeling* page 262:

> Essentially, the NEC takes into account the prevalence of the event, model performance, and the costs and scales the total cost to be between 0 and 1. Note that this approach only assigns costs to the two types of errors and might not be appropriate for problems where there are other costs or benefits.

$NEC = PCF \times (1-TP) + (1-PCF) \times FP$

## The Proposed Solution

## The Fail

## Up Next

[Naive baseline models](https://julielinx.github.io/blog/25_baseline_compare/))

### Resources

- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)



