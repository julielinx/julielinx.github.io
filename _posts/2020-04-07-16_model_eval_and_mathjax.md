---
title: "Entry 16: Model Evaluation"
categories:
  - Blog
tags:
  - model-eval
---

Max Kuhn and Kjell Johnson make the case in *[Applied Predictive Modeling](https://www.amazon.com/Applied-Predictive-Modeling-Max-Kuhn-ebook/dp/B00K15TZU0)* that models can easily overemphasize patterns that are not reproducible.

## The Problem

Unreproducible patterns can happen for a multitude of reasons: 
- Temporary localized patterns
- Spurious relationships
- Flukes in the data

Regardless of the reason, without a methodological way to evaluate the model I won't know about the unreproducible patterns until the next set of predictions. In industry, the next set of predictions could easily be in a production environment - not the place to be making avoidable mistakes if I can help it.

### What this problem is not

I do not plan on addressing other issues that can hinder model effectiveness in this series of entries. These types of issues include:

1.) Small number of observations
  - If there are too few observations in the dataset, the model may not be able to find patterns. A contributing factor to this issue could be the number of features. If there are more features than observations, this can cause problems.
  
2.) Unreliable ground truth
  - **Ground truth** is the labelling of the feature to be predicted.
  - The accuracy of this labelling can be affected by many factors. These include:
    - **Undetected**. My grandfather loved to garden. He kept these elaborate diagrams of what plants were in his yard. If I turned that diagram into a dataset and Gramps got some volunteer plants or weeds he didn't notice (volunteers are when an existing plant sprouts baby plants, ie you don't have to go buy and plant them) then the new plants would be undetected. They wouldn't end up in my dataset because we didn't know about them.
    - **Mislabelling**. Medical conditions are a good example of this. I have a medical condition called Postural Orthostatic Tachycardia Syndrome (POTS). Due to the most prevalent symptom, high heart rate, this condition is commonly misdiagnosed as anxiety. Whenever someone with POTS is diagnosed with anxiety instead of the underlying condition of POTS, that is mislabelling. (Of course medical conditions aren't always that easy, both POTS and anxiety can exist in the same person. If that's the case and only a diagnosis of anxiety is given, then POTS would qualify as undetected.)
    - **Inconsistent reporting**. Surveys or individual responses are a good example of this. I read scripts for the Nashville Film Festival's script writing contest a few times. Many readers sign up to help with this event. Each reader brings their own perception of whether a script is good or bad. Even after attempting to correct for differences in perspectives by focusing on specific areas like character development, originality, dialogue, and the like, readers tend to rate on different scales. One reader may rarely give out the highest rating while another gives them out frequently. This results in the same script receiving multiple classifications. The fact that the same script can be rated differently is an example of how inconsistent reporting would work across a dataset.
    
3.) Imbalanced classes
  - Imbalanced classes happen when one class is more prevalent in the target feature than another.
  - If the model always predicts the majority class it will be right the majority of the time.
  - Examples of this issue occur in many industries:
    - Fraud: fraud in insurance makes up about [5-10%](https://www.insurancefraud.org/statistics.htm) of claims. If a model predicts 'not fraud' every time it will be correct 90-95% of the time.
    - Medicine: [38.4% of men and women](https://www.cancer.gov/about-cancer/understanding/statistics) are estimated to be diagnosed with cancer at some point in their lifetime. Considering the length of a lifetime, this means that for any single test of cancer at any specific point in time over the course of their lifetime, a model predicting 'no cancer' will be correct the majority of the time.
    - Advertising: the response rate advertisers expect when conducting a marketing campaign is low - [around 10%](https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-or-average-email-response-rate-for-email-marketing/). Based on this number, a model designed to predict whether a given customer will respond to the marketing material would be correct 90% of the time if it predicted 'no response.'
    
4.) Changing patterns
  - Crime is a good example of this. As police learn the types of evidence that lead to convictions, criminals learn how to hide those types of evidence. Police then have to discover new types of evidence.
  - Adversarial neural networks are another example. A common use of adversarial networks is in computer vision. One neural network is responsible for identifying images. A second neural network is responsible for generating images that can be interpreted by humans, but not by the other neural networks.

Once I can see how a model is performing on the data it's given, then I can work on improving the quality of the data it gets and how it uses that data.

I also don't plan on addressing model monitoring in this series of entries. This includes issues like:

- Model drift / staleness
- Lift / performance benefits

## The Options

There are a plethora of options when it comes to evaluating model performance. There are also several best practice methods that need to be incorporated into every evaluation. These steps include:
- K-fold cross-validation
- Stratification
- Pre-processing pipeline
- Training the model
- Evaluating the model
  - Regression
    - Mean Square Error (MSE)
    - Root Mean Square Error (RMSE)
    - Coefficient of Determination ($$R^2$$)
    - Residuals
    - Sum of Square Errors (SSE)
  - Classification
    - No information rate / baseline comparisons
      - Random
      - Majority class
    - Confusion matrix
      - Accuracy
      - Error Rate
      - Precision
      - False discovery rate
      - False omission rate
      - Negative predicted value
      - Kappa statistic (Cohen's Kappa)
      - Sensitivity / True positive rate / Recall
      - False positive rate / Fall-out
      - Specificity / True negative rate
      - False negative rate
      - Youden's J Index
      - $F_{1}$ score
    - Effect of threshold
      - Receiver Operating Characteristic (ROC)
      - Area Under the Curve (AUC)
- Effect of training size
- Effect of hyperparameters
- Fine tuning parameters
- Re-evaluating the model

    
**Chart of metrics derived from the confusion matrix [source wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)**

![Chart of metrics](https://github.com/julielinx/datascience_diaries/blob/master/img/metric_explanatory_chart.PNG?raw=true)

## The Proposed Solution

I'll tackle each of these issues one at a time like I did with the series of entries about Defining the Problem.

Several of these items are processing/pipeline challenges. Once I've worked out how to do cross-validation, what kinds of stratification are available, and the best way to put them (plus pre-processing) into a pipeline, those items will be complete.

The other items, ie the evaluation metrics, I plan to run on multiple datasets to help develop a more intuitive understanding of the metric, when it's helpful, and exactly what it's telling me.

While I plan on working on visualizing the effects of training size and hyperparameters, I don't plan on addressing changing those parameters or doing any parameter fine turning / re-evaluation of the results. Improving the effectiveness of model predictions will be it's own series of entries. I anticipate deep dives into different algorithms will be part of this process, which will be too extensive to include in this series of entries.

## The Fail

### The problem

At first I tried to jump right into the evaluation steps. However, I found myself flailing around, having to explain where each step fell in the process and why I was doing it at *this* point, then changing around the order all over again. This outline entry allowed me to consolidate my thoughts, record background and context, and create a clear outline to follow for the series.S

### The post

Originally, this post didn't display correctly because Jekyll doesn't automatically interpret LaTeX, which I use pretty extensively for mathematical equations in later posts. I had to pause to integrate MathJax into my site.

It didn't end up being all that hard, finding time to sit down and figure it out was the hardest part.

## Next Up

[Data splitting and resampling](https://julielinx.github.io/blog/17_resampling/)

### Resources

- [How to use MathJax in Jekyll generated Github pages](https://haixing-hu.github.io/programming/2013/09/20/how-to-use-mathjax-in-jekyll-generated-github-pages/)
- [Use MathJax to write Equations in Jekyll blogs](http://zjuwhw.github.io/2017/06/04/MathJax.html)
- [Math on GitHub Pages](https://g14n.info/2014/09/math-on-github-pages/)
- 