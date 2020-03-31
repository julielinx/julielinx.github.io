# Entry 16 - Model Evaluation

## The Problem

Max Kuhn and Kjell Johnson make the case in *Applied Predictive Modeling*: models can easily overemphasize patterns that are not reproducible.

This can happen for a multitude of reasons: 
- temporary localized patterns
- spurious relationships
- flukes in the data

Regardless of the reason, without a methodological way to evaluate the model I won't know about the unreproducible patterns until the next set of predictions. In industry, the next set of predictions could easily be in a production environment - not the place to be making avoidable mistakes if I can help it.

### What this problem is not

I do not plan on addressing other issues can hinder model effectiveness in this series of entries. These types of issues include:

1. Small number of observations
  - If there are too few observations in the dataset, the model may not be able to find patterns. A contributing factor to this issue could be the number of features. If there are more features than observations, this can cause problems.
2. Unreliable ground truth
  -  **Ground truth** is the labelling of the feature to be predicted.
  - The accuracy of this labelling can be effected by many factors. These include:
    - **Undetected**. The medical field is a good example of undetected values. An illness or condition can go undetected leaving what should have been labelled as 'yes' instead labelled 'no.'
    - **Mislabelled**. Marketing lead generation is a good example of misidentification. A customer response could be misinterrpreted as wanting followup when no followup was desired. Or there could be entry error where the wrong key is pressed, again mislabelling the data point.
    - **Inconsistent reporting**. Surveys are a good example of inconsistent reporting. A different person is responding to each survey. These individuals bring different perspectives to the questions. This can cause the same true/false condition to be intrepreted and answered differently.
3. Imbalanced classes
  - Imbalanced classes happen when one class is more prevalent in the target feature than another.
  - If the model predicts the majority class it will be right the majority of the time.
  - Examples of this issue occur in many industries:
    - Fraud: fraud in insurance makes up about [5-10%](https://www.insurancefraud.org/statistics.htm) of claims. If a model predicts 'not fraud' every time it will be correct 90-95% of the time.
    - Medicine: [38.4% of men and women](https://www.cancer.gov/about-cancer/understanding/statistics) are estimated to be diagnosed with cancer at some point in their lifetime. Considering the length of a lifetime, this means that for any single test of cancer, a model predicting 'no cancer' will be correct the majority of the time.
    - Advertising: the response rate advertisers expect when conducting a marketing campaign is low - [around 10%](https://www.campaignmonitor.com/resources/knowledge-base/what-is-a-good-or-average-email-response-rate-for-email-marketing/). Based on this number, a model designed to predict whether a given customer will respond to the marketing material would be correct 90% of the time if it predicted 'no response.'
4. Changing patterns
  - Crime is a good example of this. As police learn the types of evidence that lead to convictions, criminals learn how to hide those types of evidence. Police then have to discover new types of evidence.
  - Adversarial neural networks are another example. A common use of adversarial networks is in computer vision. One neural network is responsible for identifying images. A second neural network is responsible for generating images that can be intrepreted by humans, but not by the first neural network.

After I can see how a model is performing on the data it's given, then I can work on improving the quality of the data it gets.

I also don't plan on addressing model moitoring in this series of entries. This includes issues like:

- Model drift / staleness
- Lift / performance benefits

## The Options

There are a plethora of options when it comes to evaluating model performance. There are also several best practice methods that need to be incorporated into every evaluation. These steps include:
- K-fold cross-validation
- Stratification
- Preprocessing pipeline
- Train model
- Evaluate model
  - Regression
    - Mean Square Error (MSE)
    - Root Mean Square Error (RMSE)
    - Coefficient of Determination (R^2)
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
      - False Positive Rate / Fall-out
      - Specificity / True negative rate
      - False negative rate
      - Youden's J Index
      - F1 score
    - Effect of threshold
      - Reciever Operating Characteristic (ROC)
      - Area Under the Curve (AUC)
- Effect of training size
- Effect of hyperparameters
- Fine tune parameters
- Re-evaluate model

    
**Chart of metrics derived from confusion matrix (source wikipedia)**
[<img src="../img/metric_explanatory_chart.png">](https://en.wikipedia.org/wiki/Confusion_matrix)

## The Proposed Solution

I'll tackle each of these issues one at a time like I did with the series of entries about Defining the Problem.

Several of these items are processing/pipeline challenges. Once I've worked out how to do cross-validation, what kinds of stratification are available, and the best way to put them, plus preprocessing, into a pipeline those items will be complete.

Other items, like all the evaluation metrics, I plan to run on multiple datasets to help develop a more intuitive understanding of the metric, when it's helpful, and exactly what it's telling me.

While I plan on working on visualizing the effects of training size and hyperparameters, I don't plan on addressing changing those parameters or doing any parameter fine turning / re-evaluation of the results. Improving the effectiveness of model predictions will be it's own series of entries. I anticipate deep dives into different algorithms will be part of this process, which will be too extensive to include in this series.

## The Fail

At first I tried to jump right in to the evaluation steps. However, I found myself flailing around, having to explain where each step fell in the process and why I was doing it at *this* point, then changing around the order too often. This outline entry allowed me to consolidate my thoughts, record background and context, and create a clear outline to follow for the series.

## Next Up

Data splitting and resampling.


```python

```
