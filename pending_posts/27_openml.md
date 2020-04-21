---
title: "Entry 27: Figuring out openml.org"
categories:
  - Blog
tags:
  - load data
---

Figuring out how to get the data from openml.org into the <font color='red'>Entry 25e notebook</font> was surprisingly difficult. All the datasets are saved as arff files, an ASCII text file type developed for use with Weka. My tool of choice, `pandas`, doesn't have a native way to load arrf files.

## The Problem

When it comes to data science, data is required.

I've been mostly using the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php), which has a lot of classic datasets. There are 497 datasets in total and the sortable list of them includes information like whether it's a classification, regression, or clustering project and what types of attributes the dataset has (categorical, integer, numerical, real).

However, I've found that the information tends to be a little spotty. I have to open each dataset within a category and look for something that will meet my needs (ie when the attribute type says integer, there may infact be some categorical features. Or if I want a binary classification problem I have to download promising looking classification datasets and check how many classes there are).

On top of that, there isn't any standardization. The data is stored, at least mostly, the way it was submitted. I've seen file types ranging from csvs to arffs. Some separate train and test sets, others put everything in the same file. And by everything, I mean everything, including the explanations of what the features are.

Most of these challenges can be overcome using `pandas`. All issues can be addressed in one way or another in Python. However, it can get time consuming to parse the data into the shape I need.

## The Options

An alternative to the UCI Machine Learning Repository is [openml.org](https://www.openml.org/search?type=data). It has around 2,500 unique datasets (both UCI and [openml.org](https://www.openml.org/search?type=data) have multiple versions of several datasets. Most of the versions have some kind of pre-processing applied to them to help facilitate quick loading of the data or training of a model).

The only problem with [openml.org](https://www.openml.org/search?type=data) is that it saves all the data as arff files. I found three ways to read arff files.

- The `arff` module in the `scipy` package
- The `openml` package
- The `fetch_openml` module in the `sklearn.datasets` package

### `arff` module

I only did a cursory exploration of the `arff` module in `scipy`. It isn't designed specifically to work with [openml.org](https://www.openml.org/search?type=data), just any arff file. So while I can load it into the notebook, I still have to download the dataset. As noted previously, my goal is to use datasets programmatically available online to avoid using up my storage space and to make the notebooks more easily reproducible (ie no manual steps of downloading the data, then finding and updating the file path).

### `openml` package

The `openml` package is designed to work directly with [openml.org](https://www.openml.org/search?type=data). There is a website for [documentation](https://openml.github.io/openml-python/master/), however I wasn't able to find a list of available functions and explanations of what they do (I finally found it [here](https://openml.github.io/OpenML/Python-API/) while completing the writeup).

Jupyter to the rescue. I used the `tab` complete functionality to see a list of available methods. First I grabbed the full list of available datasets so I could get the name of the one I was interested in (first was Titanic). Once I had the name of the dataset, the naming schema of the methods was pretty intuitive, so it was realitively easy figure out how to download the dataset.

The major benefit of the `openml` package is the list of datasets. This list returns all sorts of information like the class size of the majority and minority class, number of classes, number of features, number of observations, number of missing values (both by count of observations with missing values and an overall count of all missing values), number of numeric features, and number of symbolic features.

This information is extrememly helpful for finding datasets that allow me to target specific machine learning topics. For example, armed with the number of classifications and the majority/minority counts, I was able to pull a list of imbalanced binary classification datasets. These datasets will allow me to test strategies that will replicate the type of imbalanced data I use at work. Another example would be when I was trying to find datasets that were just numeric for <font color='red'>Entry 8</font> on center and scaling, or just categorical features for <font color='red'>Entries 13, 14, and 15</font>.

### `fetch_openml` module

`fetch_openml` needs the name or id of the dataset from [openml.org](https://www.openml.org/search?type=data). Unfortnuately, I couldn't find either of those things listed on the dataset's page. I feel like I'd found the name and version on a specific dataset's [openml.org](https://www.openml.org/search?type=data) page before, but couldn't find it for the life of me this time.

I ended up using the `openml` package to find the names and ids (I did finally figure out that the id is listed in the dataset's URL as explained [here](https://openml.github.io/OpenML/#dataset-id-and-versions)). Once I had the name, using `fetch_openml` is ridiculously easy. I just called  `titanic = fetch_openml('titanic', version=1, as_frame=True)` and ta-da, I had data.

## The Proposed Solution

Regardless of whether I use `openml` or `fetch_openml`, the data downloaded from [openml.org](https://www.openml.org/search?type=data) always returns a dictionary. The keys of the dictionary explain where different aspects of the data are kept.

The keys are:
- `data`: returns a DataFrame of just the features
- `target`: returns a pandas series of just the target
- `frame`: returns the features and target together as a DataFrame
- `feature_names`: returns a list of the feature names
- `target_names`: returns a list the target name(s)
- `DESCR`: returns the description of the dataset as a string
- `details`: returns high level info (like the name, version, id, etc) as a dictionary
- `categories`: I'm not sure what this is for, the two datasets I pulled returned empty for this key
- `url`: returns the [openml.org](https://www.openml.org/search?type=data) webpage for the specific dataset

Both `openml` and `fetch_openml` are easy and straight forward once you have the hang of the functions and methods needed. And I have to find the name or id of the dataset before I can download it with either option. Neither of the packages seems much better than the other for the basic task of loading data. I can use either.

## The Fail

The `openml` package was originally throwing errors and whining about not supporting strings for the dataset name. I finally opened a separate notebook to figure out how to use the pacakge. When I copy and pasted the same code that had errored out in order to record the error, it suddenly worked. I have no idea what caused the error or what fixed it, but it resolved and didn't return.

### Resources

- [openml.org](https://www.openml.org/search?type=data)
- [Dataset ID and versions](https://openml.github.io/OpenML/#dataset-id-and-versions)
- [OpenML Documentation APIs](https://openml.github.io/OpenML/Python-API/)
- [OpenML Python](https://openml.github.io/openml-python/master/)
- [OpenML Documentation](https://openml.github.io/OpenML/)
- [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets.php)


```python

```
