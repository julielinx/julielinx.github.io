---
title: "Entry 11: Consolidate Pre-processing Steps"
categories:
  - Blog
tags:
  - process
  - pre-process
---

Working on the machine learning process over 10 entries left my code fragmented with different versions of key functions scattered across multiple notebooks.

To ensure my process worked, I used it on multiple datasets. The code from notebook to notebook is mostly the same, just run on different data. Per usual, the notebooks can be found on my github page:
 - [Entry 11a notebook (Computer Hardware)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/11a_nb_consolidate_preprocess.ipynb)
 - [Entry 11b notebook (QSAR Fish Toxicity)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/11b_nb_consolidate_preprocess.ipynb)
 - [Entry 11c notebook (QSAR Aquatic Toxicity)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/11c_nb_consolidate_preprocess.ipynb)
 - [Entry 11d notebook (Online News Popularity)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/11d_nb_consolidate_preprocess.ipynb)

## The Problem

I spent a significant amount of time (more than 5 hours) consolidating things into functions in [Entry 10](https://julielinx.github.io/blog/10_reorder_and_predict/). However, because of the dataset I was using (tiny training set, faked test set) I didn't have to do things like split the data into train/test. The problem to tackle in this entry was to consolidate functions from previous notebooks, write new functions to catch what I missed, and generally automate whenever possible. 

## The Options

This problem was pretty straight forward. The majority of the work to figure out what needed to be addressed was already handled in entries 6-10. I just had to find the most streamlined versions of my code.

The only real addition was creating and automating a process to split out training and test datasets as well as separating the target variable.

The biggest choices revolved around determining how inclusive to make each function. One of the lessons I've learned using streaming data with nearly 600 features at work is that there is a certain level of flexibility that's needed to address different use cases.

To help get an idea for what level of flexibility was needed, I ran the automated processes on multiple datasets. To help stay within my target timeframe, I limited myself to four datasets stored on the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php) that have only numeric features. As I add more advanced pre-processing techniques I can learn best practices for feature transformations through trial and error.

*Side note*: Wherever possible, I'll be pulling data directly from the internet so I don't have to worry about storage or going over the "recommended" (read: "mandated allowance") file size on GitHub. This may cause problems if the dataset is taken offline in the future. However, the risk of that with the UCI datasets should be minimal as the UCI Repository has been a staple source of datasets for many years. It was originally created as an ftp archive in 1987.

## The Proposed Solution

I decided on four main functions:

1. Split the data into train/test/reserve sets
2. Assess correlation and collinearity
3. Pre-process the data
4. Train the model and make predictions

1 - Most tutorials, blogs, books, etc split the data into train and test sets. I decided I want a reserve set as well to give me a final sanity check once all fine tuning has been completed.

Evaluating model performance will be the next big area to tackle (after I finish the last 2 pre-processing areas). Based on the videos in Andrew Ng's [Machine Learning course](https://www.coursera.org/learn/machine-learning/), I anticipate using k-fold cross-validation on the training set, giving me a baseline to start fine tuning. Next, will be testing the learned parameters on the test set and determining whether there is any over- or underfitting. Any major issues can be addressed at that stage. This will leave me with one final dataset that hasn't been seen by the model to evaluate performance and catch any bugs.

2 - The correlation and collinearity feature reduction step is really useful when there are a lot of features. However, three of the datasets I used have less than 10 features. Separating out the correlation/collinearity reduction step gave me the option of skipping it when there are a small number of features.

3 - At some point, I'll probably split this into multiple pre-processing steps/functions. For example, some categories will probably warrant different encoding methods (binary vs ordinal vs one-hot, etc). I'll address missing values in <font color='red'>Entry 12</font> and categorical encoding in <font color='red'>Entry 14</font>. However, these types of pre-processing start to involve assessing whether the transformation improves model performance. For now, I just need a skeleton process so that I can get to the evaluation portion.

Once I've developed a set of diagnostics to assess model effectiveness, I can play with determining the best types of advanced pre-processing for specific types of input and which transformations are preferred by which algorithms.

4 - I haven't progressed very far on running different kinds of models or using hyperparameters. This function will probably end up being split into multiple functions down the road, but the skeleton can do the linear regression training and prediction stages all together.

## The Fail

### Adding a dataset

Okay, so I forgot that my pre-processing functions only take 2 datasets - training and test. This means that the reserve set wouldn't get pre-processed. Regardless of how many validation datasets I have, they all have to be pre-processed according to the parameters determined by the training dataset. As such, I just moved the second split to after the pre-processing has been complete, but before the predictions take place.

*Note from the future*, I ended up just doing train and test sets. The way I ended up setting up my process, the test set functions as a reserve set. I do cross-validation on the training set, which allows the training set to act as though it's been broken into further train and test sets. I don't get to that stage until around <font color='red'>Entry 18</font> though.

### Low correlation

The first three datasets ran through easy (so easy I almost missed the above problem). The fourth dataset started throwing errors - which is why I should always remeber to test processes and pipelines on as many diverse datasets as possible. The problem with the fourth dataset was that there were no features that had more than 0.5 correlation with the target. This resulted in an empty feature set (this can be seen in the [Entry 11d notebook (Online News Popularity)](https://github.com/julielinx/datascience_diaries/blob/master/01_ml_process/11d_nb_consolidate_preprocess.ipynb).

I thought about reverting the correlation function to pulling the X most correlated features, but I don't want to always be stuck with ~20 features or whatever arbitrary number I would have settled on. I also don't have a way to measure the efficacy of removing features, so I don't have a way to gage what number or percentage or whatever to go on. Final decision: I'm going to skip the feature reduction step for now and assess its necessity once I've got a way to evaluate model performance. I may even find a better way to do it as I continue my deep dive into machine learning concepts.

# Up Next

[Missing values](https://julielinx.github.io/blog/12_missing_values)