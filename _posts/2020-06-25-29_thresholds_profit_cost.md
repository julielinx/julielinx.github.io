---
title: "Entry 29: Thresholds - Profit and cost"
categories:
  - Blog
tags:
  - model-eval
  - dataset titanic
  - thresholds
---

Sometimes considerations other than model performance need to be accounted for when choosing a threshold.

The notebooks where I did my code for this entry can be found on my github page:
 - [Entry 29a notebook (calculations)](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/29a_nb_thresholds_profit_cost_calcs.ipynb)
 - [Entry 29b notebook (charts)](https://github.com/julielinx/datascience_diaries/blob/master/02_model_eval/29b_nb_thresholds_profit_cost_plots.ipynb)

## The Problem

When working with real world data and production applications, the correctness of the model isn't the only consideration. An example of a consideration external to the model is personnel resource allocation.

An example of this situation would be a company using lead generations like in [Entry 28](https://julielinx.github.io/blog/28_gain_lift/). There are only so many sales staff to make calls, so to calculate the best threshold, personnel constraints need to be included in the calculation.

These types of situations are where the profit calculation comes in handy.

## The Options

### Profit

$profit = xTP - yFP - zFN$

This was thrown into *[Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)* as an example of a non-accuracy based criteria.  It allocates a gain from true positives (TP) and costs for false positives (FP) and false negatives (FN) to assign a dollar amount. The example in the book was for a direct mail campaign. `x` amount was expected to be gained by customers that responded to the mailer, `y` was spent on each mailer, and `z` was the amount lost for mailers not sent to customers that would have responded.

To put hard numbers to it:

- Customers that responded to the mailer spent \$50 on average
- Each mailer costs \$1.50 to print and mail

As such, our variables would be:

- `x`: amount expected to be gained = the average amount spent less the cost of printing and mailing $=> 50 - 1.50 = 48.50$
- `y`: cost of incorrectly predicting a customer will respond = the cost of printing and mailing $=> 1.50$
- `z`: amount lost for customers that would have responded if mailer sent = the average amount spent $=> 50$
- profit $= 48.50TP - 1.50FP - 50FN$

The same basic equation can be used in use cases such as the lead generation problem or fraud. In the case of fraud, it would be viewed from a cost savings perspective instead of a profit generating perspective. `x` would be the expected savings for each case of fraud successfully identified and stopped, `y` would be costs like customers lost due to increased inconvenience and fraud specialist salaries, and `z` would be the amount lost for each fraudulent case gone undetected.

Variations on this concept are the probability cost function and normalized expected cost.

### Probability cost function (PCF)

*Applied Predictive Modeling* page 262:

> The PCF is the proportion of the total costs associated with a false-positive sample.

$PCF = \frac{P \times C(fn)}{P \times C(fp) + (1 - P) \times C(fn)}$

Where:

- *P* is the (prior) probability of the event (all positives)
  + I.E., *P* is the proportion of positives in the data
  + As such, 1 - *P* is the probability of a non-event, or the proportion of all negatives in the data
- *C(fn)* is the cost of a false negative (positive observation predicted as a negative)
- *C(fp)* is the cost of a false positive (negative observation predicted as a positive)

When implemented, the probability cost function returns a single value. This is because it uses the probability of the event - i.e., how often the actual event of interest happens. In other words, it uses the observed values, not the predictions. It then combines the information about the observed values with the costs of false negatives and false positives.

The single value output is then the proportion of the total cost that is associated with the false positives. As such, given the following information I would expect to be able to calculate the costs associated with false positives:

- \$500 was spent on the project
- The PCF was 0.36

So the cost of false positives should be $500 * 0.36 = 180$.

### Normalized expected cost (NEC)

*Applied Predictive Modeling* page 262:

> Essentially, the NEC takes into account the prevalence of the event, model performance, and the costs and scales the total cost to be between 0 and 1. Note that this approach only assigns costs to the two types of errors and might not be appropriate for problems where there are other costs or benefits.

$NEC = PCF \times (1-TP) + (1-PCF) \times FP$

## The Proposed Solution

Of the three equations, the profit calculation is the easiest to calculate and understand. This makes it the best option when working with non-mathematicians. It also has a nice upside down U shaped curve that makes the most appropriate range of values very clear.

As for the other two options, see The Fail section.

## The Fail

I'm not sure where the normalization portion of the NEC equation is. When I implemented the equation using real data, the scale wasn't between 0 and 1 as expected. In addition, if the PCF is the portion of expenses associated with the false positives, why is FP multiplied by (1 - PFC)? Shouldn't FP be multiplied by PCF?

And what's the purpose of (1-TP)? Based on previous definitions of TP in the book, it's the count of true positives in the data. But in that case (1-TP) would return the number of true positives less one as a negative value. It would make more sense if these were rates instead of counts. But I don't see anything to indicate this change in definition.

I went searching for more information online. Chris Drummond and Robert Holte appear to be the experts on NEC, as the four papers I found on it were all written by the two of them. However, the papers are all written in mathematical notation without clarifying examples and include things like:

$\frac{TP_{1} - TP_{2}}{FP_{1} - FP_{2}} = \frac{p(-)C(+|-)}{p(+)C(-|+)}$

Where:
- $p(a)$: the probability of a given example being in class $a$
- $C(a|b)$: the cost incurred if an example in class $b$ is misclassified as being in class $a$

I'm okay with the first portion of the equation, it's just the rise over the run where the points are ($FP_{1}, TP_{1}$) and ($FP_{2}, TP_{2}$). However, in the second part of the equation there is no $p(a)$ or $C(a|b)$. Now, I can guess than $a$ and $b$ are the positive and negative classes, in which case the second part of the equation translates to:

$\frac{p(a)C(b|a)}{p(b)C(a|b)}$

But it doesn't actually say that. And it's hard to confirm that guess without concrete examples or digging into the math myself. So, I'm going to shelf NEC until I have time for (and feel like) reading 10 - 36 page academic mathematics papers. After all, the goal of these entries is to break things down into 1-2 hour bite size pieces.

## Up Next

[Learning curves](https://julielinx.github.io/blog/30_learning_curves_imp_perform/)

### Resources

- [Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)
- [Explicitly Representing Expected Cost: An Alternative to ROC Representation](http://www.csi.uottawa.ca/~cdrummon/pubs/KDD00.pdf)
