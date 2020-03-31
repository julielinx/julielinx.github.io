# Entry 13 - Categorical Preliminaries

Real world data often includes categorical variables like:
- Male or female
- Continent: North America, South America, Europe, Asia, Australia, Antartica, Africa
- Smoker or Non-smoker

## The Problem<img src="../img/reading_list.jpg" width=300 align='right' style='margin:6px 6px'>

I combed through ten of my data science/machine learning/data mining books for information on categorical variables. Some assumed all data was numerical, others said categorical variables would need to be changed to numbers (without specifying how to go about that), and others offered two options for encoding (label/ordinal encoding or one hot encoding - discussed later).

Few of them went beyond explaining nominal vs ordinal. Due to the dearth of information in the standard ML books, the problem addressed here is consolidating foundational information on categorical variables into one concise entry.

## The Options

There was only one book in my arsenal that held the level of detail I was looking for on categorical variables (and I almost forgot I had it): *[Categorical Data Analysis](https://www.amazon.com/gp/product/0470463635/ref=dbs_a_def_rwt_bibl_vppi_i4)* by Alan Agresti.

There are also quite a few resources referenced on the [category-encoders pypi page](https://pypi.org/project/category-encoders/), several of which were very informative (see Resources section for links).

## The Proposed Solution

### Types of Variables

As long as I'm defining variables types, I may as well include numerical ones so they're all in the same spot.

#### Numerical

- **Discrete** values are numbers within a countable range
  - These are usually whole numbers
  - Ex: 1, 2, 3, 4
- **Continuous** variables can have any number of values
  - While measurement precision makes all values discrete in practice, continuous variables are usually those that have a large number of values with a theoritically unlimited number of possiblities between any two numbers
  - Ex: Height - 5'4", 6'1.75", 5'10.6456"


#### Categorical

- **Binary** variables have only two categories
  - Value can only be one of two values
  - Ex: Yes or No
- **Nominal** categorical values have no intristic value
  - Each value is discrete with no intristic ordering
  - Ex: apple, orange, banana (there is no way to subtract banana from the median value)
- **Ordinal** categorical values have some kind of intristic order
  - Nearby values are more similar than distant values
  - Ex: low, medium, high
  - Ex: 0-100, 101-150, 151-200
- **Interval or ratio** categorical values have numerical distance between the them
  - These are categories that are expressed numerically
  - Ex: blood pressure, temperature
  
Fun fact from *Categorical Data Analysis*, sometimes categorical variables can be measured and recorded using each of the four ways. For example, education can be binary (Obrained GED: Yes, No), nominal (type of schooling: public school, private school, home schooled), ordinal (highest degree obtained: none, high school, bachelor's, master's, doctorate), or interval (years of education completed: 1, 2, ...).

#### Categorical statistical method hierarchy

Statistical methods that can be used for each of the types apply in a hierarchy. Nominal is lowest, ordinal the middle, and interval the highest. Any method that can be used on a lower type can be used on a higher type, but additional methods that can be used on higher types cannot be used on lower types. IE: what applies to nominal variables can be applied to ordinal and interval, but a method that applies to ordinal cannot be used on nominal. For example, intervals can be added and subtracted (the difference in years of education between someone that graduated high school (12) and someone who graduated college (16 years) is 4), but nominal values cannot (yellow minus green makes no sense).

Since binary variables have only two values there is no distinction between ordered and unordered, and so methods for nominal and ordinal generally reduce to equivalent methods.

### Characteristics

Numeric variables are generally **quantative** - they are measures of values or counts. As Alan puts it 'distinct levels have differing amounts of the characteristic of interest.' This also holds true for intervals - there is an assumption of an underlying continuous characteristic.

Nominal variables are **qualitative** - distinct categories differ in quality or type, not in quantity.

Per Alan, most analysts treat ordinal variables as qualitative variables, like their nominal categorical siblings. However, he also states that they can more closely resemble interval variables if there is an assumption of an inherent underlying continuous characteristic.

### Encoding Techniques

There are quite a few different ways to encode categorical data. Below I list the six most common and intuitive ways with definitions and examples.

The seventh item lists additional methods that are available in the category-encoders module. The explanations for these methods are mathy. Trying out these more obscure encoding methods can wait until after a diagnositc suite has been built to evaluate model performance.
  
- **Label Encoding** (ie: making them ordinal variables)
  - Converts each label/category into a number
  - Ex: apple = 1, orange = 2, banana = 3
    |name|fruit|code|
    |-|-|-|
    |Fuji|apple|1|
    |Chiquita|banana|3|
    |Valencia|orange|2|
- **One Hot Encoding** (ie: turning each category into binary variables)
  - Each category is given it's own column with a 0 or 1 value
  - Ex:
    |name|apple|orange|banana|
    |-|-|-|-|
    |Fuji|1|0|0|
    |Chiquita|0|0|1|
    |Valencia|0|1|0|
- **Dummy Encoding** (ie: turning categories into binary variables, but leaving one out)
  - Each category is given its own column with a 0 or 1 value, but the last column is left off
  - This is used when each row has only one of the categories (thus the sum of the one hot encoded values for a row would be 1). Because the value of the last column can be derived from the other columns, it is left out.
  - Ex:
    |name|apple|orange|
    |-|-|-|
    |Fuji|1|0|
    |Chiquita|0|0|
    |Valencia|0|1|
- **Frequency Encoding** (ie: turning the categories into numerical variables)
  - Coverts each category to its frequency (count)
  - Ex
    |name|fruit|code|
    |-|-|-|
    |Fuji|apple|10|
    |Ambrosia|apple|10|
    |Winesap|apple|10|
    |Honeycrisp|apple|10|
    |Golden Delicious|apple|10|
    |Pink Pearl|apple|10|
    |Cortland|apple|10|
    |Washington|apple|10|
    |Gala|apple|10|
    |Granny Smith|apple|10|
    |Red|banana|5|
    |Cavendish|banana|5|
    |Latundan|banana|5|
    |Lady Finger|banana|5|
    |Saba|banana|5|
    |Valencia|orange|2|
    |Blood|orange|2|
- **Target/Impact/Likelihood Encoding** (ie: turning the categories into numerical variables)
  - Using the target variable to encode the categories
  - There are multiple ways of doing this
    - Calculate the mean target value for each category (for regression tasks)
    - Calculate the likelihood of a data point to belong to one of the classes (for classification tasks)
  - **Note**: this method requires regularization after encoding, just like other numeric variables
  - Likelihood ex:
    |name|fruit|code|red(target)|
    |-|-|-|-|
    |Fuji|apple|0.7|1|
    |Ambrosia|apple|0.7|0|
    |Winesap|apple|0.7|1|
    |Honeycrisp|apple|0.7|1|
    |Golden Delicious|apple|0.7|0|
    |Pink Pearl|apple|0.7|1|
    |Cortland|apple|0.7|1|
    |Washington|apple|0.7|1|
    |Gala|apple|0.7|1|
    |Granny Smith|apple|0.7|0|
    |Red|banana|0.2|1|
    |Cavendish|banana|0.2|0|
    |Latundan|banana|0.2|0|
    |Lady Finger|banana|0.2|0|
    |Saba|banana|0.2|0|
    |Valencia|orange|0.5|0|
    |Blood|orange|0.5|1|
- **Binary Encoding**
  - Use binary code to express the category as an ordinal and split the digits of the binary into separate columns.
  - Encodes the data in fewer dimensions that one-hot (as long as you have more categories than the length of the binary), but with some distortion
  - Ex: 
    - *Prelim steps*:
    |name|fruit|ordinal|binary|
    |-|-|-|-|
    |Fuji|apple|1|0001|
    |Chiquita|banana|3|0011|
    |Valencia|orange|2|0010|
    - *Binary Encoding* 
    |name|fruit|binary_1|binary_2|binary_3|binary_4|
    |-|-|-|-|-|-|
    |Fuji|apple|0|0|0|1|
    |Chiquita|banana|0|0|1|1|
    |Valencia|orange|0|0|1|0|
- **Other**
  - Orthogonal Polynomial Encoding
  - Forward and Reverse Helmert Encoding
  - Forward and Reverse Difference Encoding
  - Hashing
  - Sum Encoding
  - Weight of Evidence

## The Fail

I wanted to list out all the different methods listed in the category-encoders module with definitions and examples, since this is the most comprehensive list of methods I found. However, time constraints limited the amount of time I was willing to devote to understanding the provided explanations and math, especially since I don't currently have a way to evaluate how well they perform.

## Resources
- [Beyond One Hot: an exploration of categorical variables](http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/)
- [Overview of Encoding Methodologies](https://www.datacamp.com/community/tutorials/encoding-methodologies)
- Categorical Data Analysis by Alan Agresti
  - [Hardcopy on Amazon](https://www.amazon.com/gp/product/0470463635/ref=dbs_a_def_rwt_bibl_vppi_i4)
  - [PDF](https://mybiostats.files.wordpress.com/2015/03/3rd-ed-alan_agresti_categorical_data_analysis.pdf)
- [category-encoders pypi page](https://pypi.org/project/category-encoders/)
- [Statistics: The Art and Science of Learning from Data](https://www.libs.uga.edu/reserves/docs/main-spring2017/lutz-stat6220/agresti%20&%20franklin%203e.pdf) by alan Agresti and Christine Franklin


```python

```
