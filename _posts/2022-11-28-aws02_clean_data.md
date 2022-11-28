---
title: "Entry SM02: Clean Data"
categories:
  - Blog
tags:
  - aws
  - sagemaker
  - data cleaning
  - data wrangling
  - production pipeline
---

Wrangling data into a usable form is a big part of any real world machine learning problem. When tackling these types of problems, several things are generally true:

First, the information to understand and/or solve the proposed business problem is usually stored in multiple locations or files. Whether it's multiple excel files, different tables within the same database, or completely different systems, having to pull information from multiple sources and then combine them is common.

Second, the data is usually in its raw form. This means the data will need to be featurized. Free text needs to be turned into some kind of numeric representation, categories either need to be binned or encoded, numeric features may need to be standardized, lists of values turned into single number representations, or other custom transformations may need to be applied.

I chose the [Insurance Company Benchmark (COIL 2000) dataset](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29) because it let's me demonstrate some of these real world problems:

1. It's stored in multiple files
2. It has both numeric and categorical features
3. It's a binary classification problem

### Objectives

Like a production level problem, this dataset needs to be combined and put into a usable form. Here are the data wrangling steps I'll complete in this post:

- Combine train, test, and label datasets into a single full dataset
    - Production datasets don't usually come pre-split
    - Figuring out when, where, and how to split the data is an important production consideration
- Update column names to something human readable
    - This makes my life easier when completing my exploratory data analysis to understand what the data is
    - Blind data processing is generally a bad idea
- Return categorical features to their textual representation
    - Allows me to demonstrate both numeric and categorical handling as I move through creating a SageMaker Pipeline

## Prerequisites

All prerequisites from the previous post, `1_read_from_s3.ipynb`, still apply (SageMaker environment and applicable IAM roles are set up). The data that was loaded into S3 in the previous post will also be needed. Below is the code to pull the data from the [UCI page](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29) and load it to S3 incase you deleted the files from the last post.


```python
# Only run this if you deleted the output from the previous post or didn't run it at all
# If you run it accidentally, oh well, it'll just overwrite the previous files
import pandas as pd
import sagemaker

session = sagemaker.session.Session()
bucket = session.default_bucket()
prefix = '1_ins_dataset/raw'

train_uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticdata2000.txt'
test_uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/ticeval2000.txt'
gt_uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/tictgts2000.txt'
cols_uri = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-mld/dictionary.txt'

train = pd.read_table(train_uri, header=None)
test = pd.read_table(test_uri, header=None)
ground_truth = pd.read_table(gt_uri, header=None)
columns = pd.read_table(cols_uri, encoding='latin-1')

train.to_csv(f's3://{bucket}/{prefix}/train.csv', index=False)
test.to_csv(f's3://{bucket}/{prefix}/test.csv', index=False)
ground_truth.to_csv(f's3://{bucket}/{prefix}/gt.csv', index=False)
columns.to_csv(f's3://{bucket}/{prefix}/metadata/col_info.csv', index=False)
```


## Set AWS variables

This step needs to be completed pretty much any time I run anything in a SageMaker notebook. These variables are what allow me to talk to AWS resources and establish I have premission to use said resources.


```python
import sagemaker
import boto3
import pandas as pd

session = sagemaker.session.Session()
bucket = session.default_bucket()
prefix = '1_ins_dataset/raw'
```

## Read in Data

First I read in all the data from S3. After trying multiple options in the last post, I decided I prefer using `boto3.resource`. For context on how this function was created, including information on why I used the `objects.filter` and `split` functions as well as why I put the data into a dictionary, please read the pervious post.


```python
def read_mult_txt(bucket, prefix):
    s3_resource = boto3.resource("s3")
    s3_bucket = s3_resource.Bucket(bucket)

    files = {}
    for object_summary in s3_bucket.objects.filter(Prefix=prefix):
        if (len(object_summary.key.rsplit('.')) == 2) & (len(object_summary.key.split('/')) <= 3):
            files[object_summary.key.split('/')[-1].split('.')[0]] = f"s3://{bucket}/{object_summary.key}"
            
    df_dict = {}
    for df_name in files.keys():
        df_dict[df_name] = pd.read_csv(files[df_name])

    return df_dict
```


```python
df_dict = read_mult_txt(bucket, prefix)
```

## Combine data

As discussed earlier, when working on a real world dataset the data isn't generally split out into test and train for you. I want to account for this real world fact in my SageMaker Pipeline, so I need to add the test data labels to the test data, then join that with the training data. This will allow me to address when and how to split my train/test/validate data for myself.

*Note*, the original split only divided the data into train and test datasets. I'll be splitting the data into train, test, and validate. This will allow me to reserve data to evaluate my final model (validate data) after training (training data) and optimizing hyperparameters (test data). Splitting the data myself allows me to make these kinds of decisions.

In order to join all of the data into a single dataset, I need to know what columns are where. The training dataset will be my template because it already holds all of the data. Standard practice is to include the target/label as either the first column or the last. I can determine what columns are what by bringing in the data dictionary and associating the column names to the data.

### Data Dictionary

Bringing in the column names from the data dictionary isn't as easy as it sounds with this particular dataset. The column names are in a text file that isn't conducive to a dataframe. By reading it in as a table, I can get something usable. However, upon pulling it in I found that there is other information in addition to the column names.


```python
pd.set_option('display.max_rows', 50)

col_info_uri = f"s3://{bucket}/{prefix}/metadata/col_info.csv"
data_info = pd.read_table(col_info_uri)
data_info
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATA DICTIONARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nr Name Description Domain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1 MOSTYPE Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2 MAANTHUI Number of houses 1  10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3 MGEMOMV Avg size household 1  6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4 MGEMLEEF Avg age see L1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>165</th>
      <td>5 f 500  999</td>
    </tr>
    <tr>
      <th>166</th>
      <td>6 f 1000  4999</td>
    </tr>
    <tr>
      <th>167</th>
      <td>7 f 5000  9999</td>
    </tr>
    <tr>
      <th>168</th>
      <td>8 f 10.000 - 19.999</td>
    </tr>
    <tr>
      <th>169</th>
      <td>9 f 20.000 - ?</td>
    </tr>
  </tbody>
</table>
<p>170 rows × 1 columns</p>
</div>



For brevity, I only returned the top and bottom five rows of data. But when examining the full set the data dictionary clearly contains 170 rows which is comprised of five different sets of data. The information includes headers for several of the datasets and a dataset name to separate the different sections of information.

To make it more usable, I'm going to clean this up by separating the datasets, splitting out columns, and giving them their appropriate headers. I manually reviewed the data for the row indexing to separate out the different datasets.

Next I split the rows into the appropriate number of columns. Fortunately, most of this can be done by splitting on the space character and limiting the number of splits. If the column names had been more than one word long, the solution would have been more complicated. However, a simple string split gets me what I need and the column names are isolated in `data_dict['feat_info']['Name']`.


```python
data_dict = {}
data_dict['feat_info'] = data_info.iloc[1:87, 0].str.split(n=2, expand=True)
data_dict['feat_info'].columns = data_info.iloc[0, 0].split(maxsplit=2)

data_dict['L0'] = data_info.iloc[89:130, 0].str.split(n=1, expand=True)
data_dict['L0'].columns = data_info.iloc[88, 0].split()

data_dict['L1'] = data_info.iloc[131:137, 0].str.split(n=1, expand=True)
data_dict['L1'].columns = ['Value', 'Bin']

data_dict['L2'] = data_info.iloc[138:148, 0].str.split(n=1, expand=True)
data_dict['L2'].columns = ['Value', 'Bin']

data_dict['L3'] = data_info.iloc[149:159, 0].str.split(n=1, expand=True)
data_dict['L3'].columns = ['Value', 'Bin']

data_dict['L4'] = data_info.iloc[160:, 0].str.split(n=1, expand=True)
data_dict['L4'].columns = ['Value', 'Bin']

for key in data_dict.keys():
    print(key)
    display(data_dict[key].head())
```

    feat_info



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nr</th>
      <th>Name</th>
      <th>Description Domain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>MOSTYPE</td>
      <td>Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>MAANTHUI</td>
      <td>Number of houses 1  10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>MGEMOMV</td>
      <td>Avg size household 1  6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>MGEMLEEF</td>
      <td>Avg age see L1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>MOSHOOFD</td>
      <td>Customer main type see L2</td>
    </tr>
  </tbody>
</table>
</div>


    L0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>1</td>
      <td>High Income, expensive child</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2</td>
      <td>Very Important Provincials</td>
    </tr>
    <tr>
      <th>91</th>
      <td>3</td>
      <td>High status seniors</td>
    </tr>
    <tr>
      <th>92</th>
      <td>4</td>
      <td>Affluent senior apartments</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5</td>
      <td>Mixed seniors</td>
    </tr>
  </tbody>
</table>
</div>


    L1



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>1</td>
      <td>20-30 years</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2</td>
      <td>30-40 years</td>
    </tr>
    <tr>
      <th>133</th>
      <td>3</td>
      <td>40-50 years</td>
    </tr>
    <tr>
      <th>134</th>
      <td>4</td>
      <td>50-60 years</td>
    </tr>
    <tr>
      <th>135</th>
      <td>5</td>
      <td>60-70 years</td>
    </tr>
  </tbody>
</table>
</div>


    L2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>1</td>
      <td>Successful hedonists</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2</td>
      <td>Driven Growers</td>
    </tr>
    <tr>
      <th>140</th>
      <td>3</td>
      <td>Average Family</td>
    </tr>
    <tr>
      <th>141</th>
      <td>4</td>
      <td>Career Loners</td>
    </tr>
    <tr>
      <th>142</th>
      <td>5</td>
      <td>Living well</td>
    </tr>
  </tbody>
</table>
</div>


    L3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>0</td>
      <td>0%</td>
    </tr>
    <tr>
      <th>150</th>
      <td>1</td>
      <td>1 - 10%</td>
    </tr>
    <tr>
      <th>151</th>
      <td>2</td>
      <td>11 - 23%</td>
    </tr>
    <tr>
      <th>152</th>
      <td>3</td>
      <td>24 - 36%</td>
    </tr>
    <tr>
      <th>153</th>
      <td>4</td>
      <td>37 - 49%</td>
    </tr>
  </tbody>
</table>
</div>


    L4



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>0</td>
      <td>f 0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>1</td>
      <td>f 1  49</td>
    </tr>
    <tr>
      <th>162</th>
      <td>2</td>
      <td>f 50  99</td>
    </tr>
    <tr>
      <th>163</th>
      <td>3</td>
      <td>f 100  199</td>
    </tr>
    <tr>
      <th>164</th>
      <td>4</td>
      <td>f 200  499</td>
    </tr>
  </tbody>
</table>
</div>


### Label column

Now I can determine which column holds the target label. This is the column I'll be trying to predict on for this machine learning problem. A peek at the first and last few rows gives me the column name and description (reminder, standard practice is to put the label as the first or last column).


```python
display(data_dict['feat_info'].head(3))
display(data_dict['feat_info'].tail(3))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nr</th>
      <th>Name</th>
      <th>Description Domain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>MOSTYPE</td>
      <td>Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>MAANTHUI</td>
      <td>Number of houses 1  10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>MGEMOMV</td>
      <td>Avg size household 1  6</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nr</th>
      <th>Name</th>
      <th>Description Domain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>84</th>
      <td>84</td>
      <td>AINBOED</td>
      <td>Number of property insurance policies</td>
    </tr>
    <tr>
      <th>85</th>
      <td>85</td>
      <td>ABYSTAND</td>
      <td>Number of social security insurance policies</td>
    </tr>
    <tr>
      <th>86</th>
      <td>86</td>
      <td>CARAVAN</td>
      <td>Number of mobile home policies 0 - 1</td>
    </tr>
  </tbody>
</table>
</div>


#### \* Quick tip \*

Did you forget what the dataframes within the dictionary are called? I forget what I call things all the time. You could spend all day scrolling around your notebook looking them back up, or you could just call the `.keys()` method. It's much easier to delete a Jupyter cell than constantly scroll around the notebook trying to find where you named something, then trying to return to where you're currently working.

If you've forgotten the name of a variable but can recall how it starts, you can type the first few letters then hit "Tab". Jupyter will auto-complete the variable name. If there is more than one variable that starts with those letters, it will give you a list to choose from.


```python
# df_dict.keys()
```




    dict_keys(['gt', 'test', 'train'])



A review of the UCI Repo page for the target variable shows the following:

> The training set contains over 5000 descriptions of customers, including the information of whether or not they have a caravan insurance policy. A test set contains 4000 customers of whom only the organisers know if they have a caravan insurance policy.

Combining this information with the column names and descriptions, I now know that the target variable is the last column `CARAVAN`, which according to its description is an indicator of whether or not the customer has policies  with the insurance company.

## Column headers

The column headers of all the datasets are just the column index. This wouldn't be a problem except that when I concatenated the target label onto the test data, it brought with it the column name `0`, which is already taken. I now have two columns labelled `0`.


```python
test_df = pd.concat([df_dict['test'], df_dict['gt']], axis=1)
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>76</th>
      <th>77</th>
      <th>78</th>
      <th>79</th>
      <th>80</th>
      <th>81</th>
      <th>82</th>
      <th>83</th>
      <th>84</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7</td>
      <td>9</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



In order to concatenate the new test dataset with the training dataset, I may as well rename the columns with their appropriate column names now. It will make concatenating the test data onto the training data much easier. This is easily done by turning the `Name` column from the `feat_info` dataset in the `data_dict` dictionary into a list, then assigning that to my dataframe columns.


```python
data_dict['feat_info']['Name'].to_list()[:5]
```




    ['MOSTYPE', 'MAANTHUI', 'MGEMOMV', 'MGEMLEEF', 'MOSHOOFD']




```python
df_dict['train'].columns = data_dict['feat_info']['Name'].to_list()
test_df.columns = data_dict['feat_info']['Name'].to_list()
```

Now the test and train dataframes have the appropriate column names assigned as headers.


```python
test_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>CARAVAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 86 columns</p>
</div>



Since the number of columns and the column headers match, it is a matter of a single line to concatenate these datasets into a single dataframe.


```python
df = pd.concat([df_dict['train'], test_df], ignore_index=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MOSTYPE</th>
      <th>MAANTHUI</th>
      <th>MGEMOMV</th>
      <th>MGEMLEEF</th>
      <th>MOSHOOFD</th>
      <th>MGODRK</th>
      <th>MGODPR</th>
      <th>MGODOV</th>
      <th>MGODGE</th>
      <th>MRELGE</th>
      <th>...</th>
      <th>APERSONG</th>
      <th>AGEZONG</th>
      <th>AWAOREG</th>
      <th>ABRAND</th>
      <th>AZEILPL</th>
      <th>APLEZIER</th>
      <th>AFIETS</th>
      <th>AINBOED</th>
      <th>ABYSTAND</th>
      <th>CARAVAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9817</th>
      <td>33</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9818</th>
      <td>24</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9819</th>
      <td>36</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9820</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9821</th>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9822 rows × 86 columns</p>
</div>



Once the data has all been joined into a single dataframe, I want to run some basic "sniff" tests on it. This allows me to be confident that there are no errors in the code that introduce incorrect data into my dataset. The only thing I'm really interested in evaluating for this merge is the number of rows and columns.

I know that the number of columns should be the same as the number of rows in the `Name` column. At 86 columns, this is correct.


```python
df.shape
```

To get the correct number of rows I first reviewed the UCI Repo page. It says there are "5000 descriptions of customers" in the training dataset and "4000 customers" in the test dataset. Which means there should be 9,000 rows in the combined dataset. This number doesn't match the number of rows listed in my dataframe, which is 9,822, almost 1,000 more rows than expected.

However, in looking at the shape of my original train and test datasets, I can see that the numbers on the UCI page for the training data was an estimate instead of an exact figure.

#### \* Quick Tip \*

It never hurts to double check. However, it sure can hurt if you don't.


```python
print('Training data has', df_dict['train'].shape[0], 'rows')
print('Test data has', df_dict['test'].shape[0], 'rows')
```

    Training data has 5822 rows
    Test data has 4000 rows
    Test data has 4000 rows


## Column names

The provided column names are very cryptic. The variables that start with 'M' in this dataset are a perfect example. Per the Dataset Information:

> Note: All the variables starting with M are zipcode variables. They give information on the distribution of that variable, e.g. Rented house, in the zipcode area of the customer.

I.E. These variables are aggregated data enrichments and don't necessarily directly reflect the customer in question. This type of information could easily lead to bias within a model. While this dataset is for direct marketing purposes, and thus sterotyping isn't necessarily a big deal, if this information were used to assess the insurability of a customer or cost of their policy, there could be direct consequences that unfairly impact customers.

Based on these types of considerations, I prefer column names that let me quickly understand what the data in a column represents.

I manually copied and pasted the column descriptions and altered them to my liking. With only 86 columns, this is a feasible solution. If I had more columns, I might not be so quick to resort to manual data entry. For convenience, I've included my column names below so the copy and paste solution doesn't have to be replicated by others.


```python
col_names = ['zip_agg_customer_subtype',
             'zip_agg_number_of_houses',
             'zip_agg_avg_size_household',
             'zip_agg_avg_age',
             'zip_agg_customer_main_type',
             'zip_agg_roman_catholic',
             'zip_agg_protestant',
             'zip_agg_other_religion',
             'zip_agg_no_religion',
             'zip_agg_married',
             'zip_agg_living_together',
             'zip_agg_other_relation',
             'zip_agg_singles',
             'zip_agg_household_without_children',
             'zip_agg_household_with_children',
             'zip_agg_high_level_education',
             'zip_agg_medium_level_education',
             'zip_agg_lower_level_education',
             'zip_agg_high_status',
             'zip_agg_entrepreneur',
             'zip_agg_farmer',
             'zip_agg_middle_management',
             'zip_agg_skilled_labourers',
             'zip_agg_unskilled_labourers',
             'zip_agg_social_class_a',
             'zip_agg_social_class_b1',
             'zip_agg_social_class_b2',
             'zip_agg_social_class_c',
             'zip_agg_social_class_d',
             'zip_agg_rented_house',
             'zip_agg_home_owners',
             'zip_agg_1_car',
             'zip_agg_2_cars',
             'zip_agg_no_car',
             'zip_agg_national_health_service',
             'zip_agg_private_health_insurance',
             'zip_agg_income_<_30.000',
             'zip_agg_income_30-45.000',
             'zip_agg_income_45-75.000',
             'zip_agg_income_75-122.000',
             'zip_agg_income_>123.000',
             'zip_agg_average_income',
             'zip_agg_purchasing_power_class',
             'contri_private_third_party_ins',
             'contri_third_party_ins_(firms)',
             'contri_third_party_ins_(agriculture)',
             'contri_car_policies',
             'contri_delivery_van_policies',
             'contri_motorcycle/scooter_policies',
             'contri_lorry_policies',
             'contri_trailer_policies',
             'contri_tractor_policies',
             'contri_agricultural_machines_policies',
             'contri_moped_policies',
             'contri_life_ins',
             'contri_private_accident_ins_policies',
             'contri_family_accidents_ins_policies',
             'contri_disability_ins_policies',
             'contri_fire_policies',
             'contri_surfboard_policies',
             'contri_boat_policies',
             'contri_bicycle_policies',
             'contri_property_ins_policies',
             'contri_ss_ins_policies',
             'nbr_private_third_party_ins',
             'nbr_third_party_ins_(firms)',
             'nbr_third_party_ins_(agriculture)',
             'nbr_car_policies',
             'nbr_delivery_van_policies',
             'nbr_motorcycle/scooter_policies',
             'nbr_lorry_policies',
             'nbr_trailer_policies',
             'nbr_tractor_policies',
             'nbr_agricultural_machines_policies',
             'nbr_moped_policies',
             'nbr_life_ins',
             'nbr_private_accident_ins_policies',
             'nbr_family_accidents_ins_policies',
             'nbr_disability_ins_policies',
             'nbr_fire_policies',
             'nbr_surfboard_policies',
             'nbr_boat_policies',
             'nbr_bicycle_policies',
             'nbr_property_ins_policies',
             'nbr_ss_ins_policies',
             'nbr_mobile_home_policies']
```


```python
df.columns = col_names
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_agg Customer Subtype</th>
      <th>zip_agg Number of houses</th>
      <th>zip_agg Avg size household</th>
      <th>zip_agg Avg age</th>
      <th>zip_agg Customer main type</th>
      <th>zip_agg Roman catholic</th>
      <th>zip_agg Protestant</th>
      <th>zip_agg Other religion</th>
      <th>zip_agg No religion</th>
      <th>zip_agg Married</th>
      <th>...</th>
      <th>Nbr private accident ins policies</th>
      <th>Nbr family accidents ins policies</th>
      <th>Nbr disability ins policies</th>
      <th>Nbr fire policies</th>
      <th>Nbr surfboard policies</th>
      <th>Nbr boat policies</th>
      <th>Nbr bicycle policies</th>
      <th>Nbr property ins policies</th>
      <th>Nbr ss ins policies</th>
      <th>Nbr mobile home policies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>10</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



## Reverse entineer categorical transformations

I chose this dataset because it had both numeric and categorical variables. I determined this by looking at the "Attribute Characteristics" as listed on the [Insurance Company Benchmark Data Set](https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+%28COIL+2000%29) page. It lists "Attribute Characteristics: Categorical, Integer."

Based on the initial examination, this dataset has obviously already been turned into numeric variables, thus negating the whole "includes categorical variables" aspect. This is one of the biggest reasons I don't generally use the UCI datasets: too many of them are already preprocessed and aren't representative of what I see in the real world. I cover more of the preprocessing already done to the dataset in the EDA post.

The data dictionary holds the key to which columns used to be categorical, it's a simple indicator at the end of the `Description Domain` column. This only applies to the features at indices 0, 3, 4, 5, and 43, as shown below.

*Note*, `Domain` should be it's own column. However, there was no super easy way to split it out and it only applies to five rows, so I didn't bother worrying about it.


```python
data_dict['feat_info'].iloc[[0, 3, 4, 5, 43], :]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nr</th>
      <th>Name</th>
      <th>Description Domain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>MOSTYPE</td>
      <td>Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>MGEMLEEF</td>
      <td>Avg age see L1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>MOSHOOFD</td>
      <td>Customer main type see L2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>MGODRK</td>
      <td>Roman catholic see L3</td>
    </tr>
    <tr>
      <th>44</th>
      <td>44</td>
      <td>PWAPART</td>
      <td>Contribution private third party insurance see L4</td>
    </tr>
  </tbody>
</table>
</div>



The `L0` - `L4` datasets that are included in the Data Dictionary are starting to make more sense. They're the textual representation of the categorical features.


```python
for key in ['L0', 'L1', 'L2', 'L3', 'L4']:
    print(key)
    display(data_dict[key].head())
```

    L0



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>1</td>
      <td>High Income, expensive child</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2</td>
      <td>Very Important Provincials</td>
    </tr>
    <tr>
      <th>91</th>
      <td>3</td>
      <td>High status seniors</td>
    </tr>
    <tr>
      <th>92</th>
      <td>4</td>
      <td>Affluent senior apartments</td>
    </tr>
    <tr>
      <th>93</th>
      <td>5</td>
      <td>Mixed seniors</td>
    </tr>
  </tbody>
</table>
</div>


    L1



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>131</th>
      <td>1</td>
      <td>20-30 years</td>
    </tr>
    <tr>
      <th>132</th>
      <td>2</td>
      <td>30-40 years</td>
    </tr>
    <tr>
      <th>133</th>
      <td>3</td>
      <td>40-50 years</td>
    </tr>
    <tr>
      <th>134</th>
      <td>4</td>
      <td>50-60 years</td>
    </tr>
    <tr>
      <th>135</th>
      <td>5</td>
      <td>60-70 years</td>
    </tr>
  </tbody>
</table>
</div>


    L2



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>1</td>
      <td>Successful hedonists</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2</td>
      <td>Driven Growers</td>
    </tr>
    <tr>
      <th>140</th>
      <td>3</td>
      <td>Average Family</td>
    </tr>
    <tr>
      <th>141</th>
      <td>4</td>
      <td>Career Loners</td>
    </tr>
    <tr>
      <th>142</th>
      <td>5</td>
      <td>Living well</td>
    </tr>
  </tbody>
</table>
</div>


    L3



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149</th>
      <td>0</td>
      <td>0%</td>
    </tr>
    <tr>
      <th>150</th>
      <td>1</td>
      <td>1 - 10%</td>
    </tr>
    <tr>
      <th>151</th>
      <td>2</td>
      <td>11 - 23%</td>
    </tr>
    <tr>
      <th>152</th>
      <td>3</td>
      <td>24 - 36%</td>
    </tr>
    <tr>
      <th>153</th>
      <td>4</td>
      <td>37 - 49%</td>
    </tr>
  </tbody>
</table>
</div>


    L4



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Value</th>
      <th>Bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>0</td>
      <td>f 0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>1</td>
      <td>f 1  49</td>
    </tr>
    <tr>
      <th>162</th>
      <td>2</td>
      <td>f 50  99</td>
    </tr>
    <tr>
      <th>163</th>
      <td>3</td>
      <td>f 100  199</td>
    </tr>
    <tr>
      <th>164</th>
      <td>4</td>
      <td>f 200  499</td>
    </tr>
  </tbody>
</table>
</div>


For brevity, I only printed the head of each of the `Lx` datasets. From this information I can determine that only two of the categorical features make sense to return to their textual representation.

`L0` and `L2` can be mapped back to their original text. I can one-hot-encode these variables in my SageMaker Pipeline. `L1`, `L3`, and `L4` are binned representations of the original values. There is very little benefit to mapping these back to the textual representation, so I'll leave them alone.

Replacing values in a dataframe can be as simple as using a dictionary to map `old value` to `new value`. I can create this dictionary using the `L0` and `L2` dataframes. The `Value` column corresponds with the numeric representation in the dataset. The `Label` and `Bin` columns correspond with the original text. The only catch was to ensure that the `Value` column was numeric (to match the datatype in the dataframe) and to set `Value` to the index so it appropriately translated as the key in the dictionary.


```python
data_dict['L0']['Value'] = pd.to_numeric(data_dict['L0']['Value'])
l0_dict = data_dict['L0'].set_index('Value').to_dict()['Label']
l0_dict
```




    {1: 'High Income, expensive child',
     2: 'Very Important Provincials',
     3: 'High status seniors',
     4: 'Affluent senior apartments',
     5: 'Mixed seniors',
     6: 'Career and childcare',
     7: "Dinki's (double income no kids)",
     8: 'Middle class families',
     9: 'Modern, complete families',
     10: 'Stable family',
     11: 'Family starters',
     12: 'Affluent young families',
     13: 'Young all american family',
     14: 'Junior cosmopolitan',
     15: 'Senior cosmopolitans',
     16: 'Students in apartments',
     17: 'Fresh masters in the city',
     18: 'Single youth',
     19: 'Suburban youth',
     20: 'Etnically diverse',
     21: 'Young urban have-nots',
     22: 'Mixed apartment dwellers',
     23: 'Young and rising',
     24: 'Young, low educated ',
     25: 'Young seniors in the city',
     26: 'Own home elderly',
     27: 'Seniors in apartments',
     28: 'Residential elderly',
     29: 'Porchless seniors: no front yard',
     30: 'Religious elderly singles',
     31: 'Low income catholics',
     32: 'Mixed seniors',
     33: 'Lower class large families',
     34: 'Large family, employed child',
     35: 'Village families',
     36: "Couples with teens 'Married with children'",
     37: 'Mixed small town dwellers',
     38: 'Traditional families',
     39: 'Large religous families',
     40: 'Large family farms',
     41: 'Mixed rurals'}




```python
data_dict['L2']['Value'] = pd.to_numeric(data_dict['L2']['Value'])
l2_dict = data_dict['L2'].set_index('Value').to_dict()['Bin']
l2_dict
```




    {1: 'Successful hedonists',
     2: 'Driven Growers',
     3: 'Average Family',
     4: 'Career Loners',
     5: 'Living well',
     6: 'Cruising Seniors',
     7: 'Retired and Religeous',
     8: 'Family with grown ups',
     9: 'Conservative families',
     10: 'Farmers'}




```python
display(data_dict['feat_info'].iloc[[0, 4], :])
print('L0:', df.columns[0])
print('L2:', df.columns[4])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Nr</th>
      <th>Name</th>
      <th>Description Domain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>MOSTYPE</td>
      <td>Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>MOSHOOFD</td>
      <td>Customer main type see L2</td>
    </tr>
  </tbody>
</table>
</div>


    L0: zip_agg Customer Subtype
    L2: zip_agg Customer main type


With the mapping dictionaries specified, all I need now is to use the `.replace()` method on the appropriate dataframe column.


```python
df[df.columns[0]] = df[df.columns[0]].replace(l0_dict)
df[df.columns[4]] = df[df.columns[4]].replace(l2_dict)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_agg Customer Subtype</th>
      <th>zip_agg Number of houses</th>
      <th>zip_agg Avg size household</th>
      <th>zip_agg Avg age</th>
      <th>zip_agg Customer main type</th>
      <th>zip_agg Roman catholic</th>
      <th>zip_agg Protestant</th>
      <th>zip_agg Other religion</th>
      <th>zip_agg No religion</th>
      <th>zip_agg Married</th>
      <th>...</th>
      <th>Nbr private accident ins policies</th>
      <th>Nbr family accidents ins policies</th>
      <th>Nbr disability ins policies</th>
      <th>Nbr fire policies</th>
      <th>Nbr surfboard policies</th>
      <th>Nbr boat policies</th>
      <th>Nbr bicycle policies</th>
      <th>Nbr property ins policies</th>
      <th>Nbr ss ins policies</th>
      <th>Nbr mobile home policies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lower class large families</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>Family with grown ups</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mixed small town dwellers</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>Family with grown ups</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mixed small town dwellers</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>Family with grown ups</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Modern, complete families</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>Average Family</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Large family farms</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>Farmers</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 86 columns</p>
</div>



## Save data

The last step is to save all my hard work back to S3.


```python
df.to_csv(f's3://{bucket}/{prefix}/full.csv', index=False)
```

## Delete Files

To ensure no ongoing charges are charged to your account, you can delete the files from S3.


```python
folder_prefix = '1_ins_dataset/'
s3_resource = boto3.resource("s3")
```


```python
s3_bucket = s3_resource.Bucket(bucket)
s3_bucket.objects.filter(Prefix=folder_prefix).delete()
```


```python
import pandas as pd
import sagemaker

session = sagemaker.session.Session()
bucket = session.default_bucket()
prefix = '1_ins_dataset/raw'

train = pd.read_csv(f's3://{bucket}/{prefix}/train.csv')
test = pd.read_csv(f's3://{bucket}/{prefix}/test.csv')
cols = pd.read_csv(f's3://{bucket}/{prefix}/col_info.csv')
```


```python
cols.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATA DICTIONARY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nr Name Description Domain</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1 MOSTYPE Customer Subtype see L0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2 MAANTHUI Number of houses 1  10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3 MGEMOMV Avg size household 1  6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4 MGEMLEEF Avg age see L1</td>
    </tr>
  </tbody>
</table>
</div>


