---
title: "Entry NLP3: Clean Data and Split into N-grams"
categories:
  - Blog
tags:
  - nlp
  - ngrams
---

In the first entry of this series, I figured out how to process the raw files. In the second entry, I figured out how to load all files in a directory (even if it has subdirectories) and store the data.

Now I'm ready to make the analysis case insensitive, remove punctuation and stopwords, and split what's left into n-grams.

*Side note:* To be fair, I worked on a pretty extensive NLP problem a few years ago. I'll be reusing code and logic from that project.

```python
import pandas as pd
import os
from IPython.display import display

import string
import re
import itertools
import nltk
nltk.download('stopwords')
```

```python
def read_script(file_path):
    corpus = ''
    with open(file_path, 'r', encoding='latin-1') as l:
        for line in l:
            if (re.match('[^\d+]', line)
               ) and (re.match('^(?!\s*$).+', line)
                      ) and not (re.match('(.*www.*)|(.*http:*)', line)
                                ) and not (re.match('Sync and correct*', line)):
                line = re.sub('</?i>|</?font.*>', '', line)
                corpus = corpus + ' ' + line
    return corpus

def load_files_to_dict(file_path, return_dict):    
    for thing in os.scandir(file_path):
        if thing.is_dir():
            new_path = os.path.join(file_path, thing.name)
            new_dict = return_dict[thing.name] = {}
            load_files_to_dict(new_path, new_dict)
        elif thing.is_file:
            return_dict[thing.name] = read_script(f'{file_path}/{thing.name}')
    return return_dict
```


```python
file_path = os.path.join(os.getcwd(), 'data', '1960s')
unilayer_dict = load_files_to_dict(file_path, {})
```

## Remove Punctuation

The list of things to remove includes `\n`, which denotes a newline. I found that including `\r` (a carriage return) and `\t` (a tab) is also helpful. These characters can all be hard to spot as they are generally invisible and can randomly attach themselves to otherwise normal words.


```python
newline_list = '\t\r\n'
```

Next I'll spell out the special characters I want to remove from the text. Fortunately, there's a list of punctuation included in the `string` library.


```python
string.punctuation
```

    '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


This list is pretty comprehensive. Between this and the `newline_list` I created above all the remaining characters from the "Remove" list are now addressed. For quick reference, I still had the following items to remove:

- '#'
- '-'
- '('
- ')'
- '"'
- '\n'

In my previous project, I discovered the `translate` method. It replaces specified characters with those described in a dictionary or mapping table. The method `maketrans` creates the mapping table. This set of methods is very handy method for proessing strings.

Now I can specify all my variables:


```python
newline_list = '\t\r\n'
remove_newline = str.maketrans(' ', ' ', newline_list)
punct_list = string.punctuation
nopunct = str.maketrans('', '', punct_list)
```

To process the data, I can then just apply `str.translate` to the column holding the text.


```python
df[text_col].fillna("").str.lower().str.translate(remove_newline).str.translate(nopunct).str.split()
```

This particular strategy hinges on the text being a value in a dataframe column. However, the output from the last notebook is a dictionary.


```python
list(unilayer_dict.keys())[:5]
```

    ['The Twilight Zone - 3x17 - One More Pallbearer.srt',
     'The Twilight Zone - 3x05 - A Game of Pool.srt',
     'The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room.srt',
     'The Twilight Zone - 4x05 - Mute.srt',
     'The Twilight Zone - 3x04 - The Passersby.srt']

```python
unilayer_dict['The Twilight Zone - 4x05 - Mute.srt'][:500]
```

    ' You unlock this door\n with the key of imagination.\n Beyond it is another dimension-\n a dimension of sound,\n a dimension of sight,\n a dimension of mind.\n You\'re moving into a land\n of both shadow and substance,\n of things and ideas.\n You\'ve just crossed over\n into the twilight zone.\n So...\n "the undersigned,\n "having accepted\n the following propositions:\n "A, that prior\n to the inception of language,\n "man communicated\n by telepathic means;\n "and b, that this ability\n not only still exists\n "but'

### Convert dictionary to dataframe

A dictionary is easily converted into a dataframe with `pd.DataFrame.from_dict`. The gotsha for this particular use case is the parameter `orient`, it has to be set to `index` in order to use key:value as rows instead of columns. Conversely, it uses the key as the index.


```python
pd.DataFrame.from_dict(unilayer_dict, orient='index').head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>The Twilight Zone - 3x17 - One More Pallbearer.srt</th>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>The Twilight Zone - 3x05 - A Game of Pool.srt</th>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room.srt</th>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>The Twilight Zone - 4x05 - Mute.srt</th>
      <td>You unlock this door\n with the key of imagin...</td>
    </tr>
    <tr>
      <th>The Twilight Zone - 3x04 - The Passersby.srt</th>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
  </tbody>
</table>
</div>


The indexing quirk is easily fixed with `reset_index` to make all my variables accessible as columns. However, the I have terrible columns names. Then I give the column names intuitive names.


```python
pd.DataFrame.from_dict(unilayer_dict, orient='index').reset_index().head()
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
      <th>index</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Twilight Zone - 3x17 - One More Pallbearer...</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Twilight Zone - 3x05 - A Game of Pool.srt</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Twilight Zone - 2x03 - Nervous Man in a Fo...</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Twilight Zone - 4x05 - Mute.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Twilight Zone - 3x04 - The Passersby.srt</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
  </tbody>
</table>
</div>


Ultimately this will all be in a single function or series of functions and the column name won't matter. However, I find it much easier to write and read the code when there are descriptive names - this goes for column names and function names. So I'm going to change the column names to be more easily understood.


```python
test = pd.DataFrame.from_dict(unilayer_dict, orient='index').reset_index().rename(columns={'index':'script_name', 0:'corpus'})
test.head()
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
      <th>script_name</th>
      <th>corpus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Twilight Zone - 3x17 - One More Pallbearer...</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Twilight Zone - 3x05 - A Game of Pool.srt</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Twilight Zone - 2x03 - Nervous Man in a Fo...</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Twilight Zone - 4x05 - Mute.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Twilight Zone - 3x04 - The Passersby.srt</td>
      <td>You're traveling\n through another dimension-...</td>
    </tr>
  </tbody>
</table>
</div>


While it is a single line of code, it is a little unwieldy and I'll need to apply it to all the dictionaries, so I'll write a quick function to do it for me.


```python
def convert_dict_df(script_dict):
    return pd.DataFrame.from_dict(script_dict, orient='index').reset_index().rename(columns={'index':'script_name', 0:'corpus'})
```

### Remove Punctuation

Now that the values are conveniently located in a dataframe, I just have to apply the logic defined earlier. To make it easier, I'll put the logic into a function, then apply that function to the example dataframe.


```python
def punct_tokens(df, text_col):
    newline_list = '\t\r\n'
    remove_newline = str.maketrans(' ', ' ', newline_list)
    punct_list = string.punctuation
    nopunct = str.maketrans('', '', punct_list)
    df['no_punct_tokens'] = df[text_col].fillna("").str.lower().str.translate(remove_newline).str.translate(nopunct).str.split()
    return df
```


```python
punct_test = punct_tokens(test, 'corpus')
punct_test.head()
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
      <th>script_name</th>
      <th>corpus</th>
      <th>no_punct_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Twilight Zone - 3x17 - One More Pallbearer...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Twilight Zone - 3x05 - A Game of Pool.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Twilight Zone - 2x03 - Nervous Man in a Fo...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Twilight Zone - 4x05 - Mute.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Twilight Zone - 3x04 - The Passersby.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
    </tr>
  </tbody>
</table>
</div>


## Remove stopwords

Now that the punctuation is out of the way, I can start thinking about the breaking the text into different sized n-grams. What has worked for me in the past is to split the string that's had punctuation removed into unigrams (called one-grams in the homework), the create different sizes of n-gram from there.

However, to get words with actual meaning, I first need to remove stopwords.

The `nltk` library has a handy list of stopwords. *Note:* Using the `nltk` library is beyond the scope of this series of entries. Historically, my use of the `nltk` libray has mostly been limited to the stopword list and n-gram creation. I have used the `FreqDist` and `ConditionalFreqDist` functions, but found them a bit tempermental and ended up coding frequency counts myself for this exercise (see the next post).


```python
nltk.corpus.stopwords.words('english')
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
     'ourselves', 'you', "you're", "you've", "you'll",
     "you'd", 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', "she's",
     'her', 'hers', 'herself', 'it', "it's", 'its',
     'itself', 'they', 'them', 'their', 'theirs',
     'themselves', 'what', 'which', 'who', 'whom',
     'this', 'that', "that'll", 'these', 'those',
     'am', 'is', 'are', 'was', 'were', 'be', 'been',
     'being', 'have', 'has', 'had', 'having', 'do',
     'does', 'did', 'doing', 'a', 'an', 'the', 'and',
     'but', 'if', 'or', 'because', 'as', 'until',
     'while', 'of', 'at', 'by', 'for', 'with', 'about',
     'against', 'between', 'into', 'through', 'during',
     'before', 'after', 'above', 'below', 'to', 'from',
     'up', 'down', 'in', 'out', 'on', 'off', 'over',
     'under', 'again', 'further', 'then', 'once',
     'here', 'there', 'when', 'where', 'why', 'how',
     'all', 'any', 'both', 'each', 'few', 'more',
     'most', 'other', 'some', 'such', 'no', 'nor',
     'not', 'only', 'own', 'same', 'so', 'than', 'too',
     'very', 's', 't', 'can', 'will', 'just', 'don',
     "don't", 'should', "should've", 'now', 'd', 'll',
     'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
     'couldn', "couldn't", 'didn', "didn't", 'doesn',
     "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
     'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
     "mightn't", 'mustn', "mustn't", 'needn', "needn't",
     'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
     "wasn't", 'weren', "weren't", 'won', "won't",
     'wouldn', "wouldn't"]


The best way I found to remove stopwords was to use list comprehension in a lambda function.

All the code for this section was re-used, so I'll lump the results all together.


```python
def create_ngrams(df):
    stop = nltk.corpus.stopwords.words('english')
    df['unigrams'] = df['no_punct_tokens'].apply(lambda x: [item for item in x if item not in stop])
    df['bigrams'] = df['unigrams'].apply(lambda x:(list(nltk.bigrams(x))))
    df['trigrams'] = df['unigrams'].apply(lambda x:(list(nltk.trigrams(x))))
    return df
```


```python
create_ngrams(punct_test).head()
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
      <th>script_name</th>
      <th>corpus</th>
      <th>no_punct_tokens</th>
      <th>unigrams</th>
      <th>bigrams</th>
      <th>trigrams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Twilight Zone - 3x17 - One More Pallbearer...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Twilight Zone - 3x05 - A Game of Pool.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Twilight Zone - 2x03 - Nervous Man in a Fo...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Twilight Zone - 4x05 - Mute.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
      <td>[unlock, door, key, imagination, beyond, anoth...</td>
      <td>[(unlock, door), (door, key), (key, imaginatio...</td>
      <td>[(unlock, door, key), (door, key, imagination)...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Twilight Zone - 3x04 - The Passersby.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
  </tbody>
</table>
</div>


I appreciate this data structure because if there is anything that doesn't make sense later in the analysis, I can search for it and track it back to the source, i.e. as long as I can find it in the designated n-gram column, I can see what the corpus looked like in the original form (the full concatenated string), after removal of punctuation, after removal of the stopwords, and converted to n-grams as well as being able to track it back to the script it came from because I have the script name.

The code to create this dataframe is a good chunk of code that's all related, so I'll combine it into a single function for easy of use.


```python
def create_ngram_df(script_dict, text_col):
    df = convert_dict_df(script_dict)
    df = punct_tokens(df, text_col)
    df = create_ngrams(df)
    return df
```


```python
authentic_ngram_df = create_ngram_df(unilayer_dict, 'corpus')
authentic_ngram_df
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
      <th>script_name</th>
      <th>corpus</th>
      <th>no_punct_tokens</th>
      <th>unigrams</th>
      <th>bigrams</th>
      <th>trigrams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Twilight Zone - 3x17 - One More Pallbearer...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>The Twilight Zone - 3x05 - A Game of Pool.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Twilight Zone - 2x03 - Nervous Man in a Fo...</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Twilight Zone - 4x05 - Mute.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
      <td>[unlock, door, key, imagination, beyond, anoth...</td>
      <td>[(unlock, door), (door, key), (key, imaginatio...</td>
      <td>[(unlock, door, key), (door, key, imagination)...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Twilight Zone - 3x04 - The Passersby.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>116</th>
      <td>The Twilight Zone - s05e36 - The Bewitchin' Po...</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
      <td>[unlock, door, key, imagination, beyond, anoth...</td>
      <td>[(unlock, door), (door, key), (key, imaginatio...</td>
      <td>[(unlock, door, key), (door, key, imagination)...</td>
    </tr>
    <tr>
      <th>117</th>
      <td>The Twilight Zone - 3x03 - The Shelter.srt</td>
      <td>You're traveling\n through another dimension-...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>The Twilight Zone - s05e21 - Spur of the Momen...</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
      <td>[unlock, door, key, imagination, beyond, anoth...</td>
      <td>[(unlock, door), (door, key), (key, imaginatio...</td>
      <td>[(unlock, door, key), (door, key, imagination)...</td>
    </tr>
    <tr>
      <th>119</th>
      <td>The Twilight Zone - 2x29 - The Obsolete Man.srt</td>
      <td>You're traveling\n through another dimension\...</td>
      <td>[youre, traveling, through, another, dimension...</td>
      <td>[youre, traveling, another, dimension, dimensi...</td>
      <td>[(youre, traveling), (traveling, another), (an...</td>
      <td>[(youre, traveling, another), (traveling, anot...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>The Twilight Zone - s05e13 - Ring-A-Ding Girl.srt</td>
      <td>You unlock this door\n with the key of imagin...</td>
      <td>[you, unlock, this, door, with, the, key, of, ...</td>
      <td>[unlock, door, key, imagination, beyond, anoth...</td>
      <td>[(unlock, door), (door, key), (key, imaginatio...</td>
      <td>[(unlock, door, key), (door, key, imagination)...</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 6 columns</p>
</div>


To handle the multiple corpora of the 21st century scripts, I retained the dictionary-holding-another-data-structure set up. The name of each grouping ('Pan-Am', 'Mad_Med', 'The_Kennedys', 'X-Men_First_Class') is a key and the dataframe is the value. Using this, I can continue to reap the benefits of my functions, while keeping the groups, and their individual analyses, separate.


```python
test_ngram_dict = {}
for script_group in list(bilayer_dict.keys()):
    test_ngram_dict[script_group] = create_ngram_df(bilayer_dict[script_group], 'corpus')
```


```python
test_ngram_dict['Pan_Am']
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
      <th>script_name</th>
      <th>corpus</th>
      <th>no_punct_tokens</th>
      <th>unigrams</th>
      <th>bigrams</th>
      <th>trigrams</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pan.Am.S01E09.srt</td>
      <td>Previously on "Pan Am"...\n Look, I get to se...</td>
      <td>[previously, on, pan, am, look, i, get, to, se...</td>
      <td>[previously, pan, look, get, see, world, sam, ...</td>
      <td>[(previously, pan), (pan, look), (look, get), ...</td>
      <td>[(previously, pan, look), (pan, look, get), (l...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pan.Am.S01E08.srt</td>
      <td>ï»¿1\n Previously on "Pan Am"...\n Let's keep...</td>
      <td>[ï»¿1, previously, on, pan, am, lets, keep, it...</td>
      <td>[ï»¿1, previously, pan, lets, keep, new, york,...</td>
      <td>[(ï»¿1, previously), (previously, pan), (pan, ...</td>
      <td>[(ï»¿1, previously, pan), (previously, pan, le...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pan.Am.S01E05.srt</td>
      <td>Previously on "Pan Am"...\n What do you think...</td>
      <td>[previously, on, pan, am, what, do, you, think...</td>
      <td>[previously, pan, think, youre, ran, away, wed...</td>
      <td>[(previously, pan), (pan, think), (think, your...</td>
      <td>[(previously, pan, think), (pan, think, youre)...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pan.Am.S01E11.srt</td>
      <td>Previously on "Pan Am".\n MI6 will want answe...</td>
      <td>[previously, on, pan, am, mi6, will, want, ans...</td>
      <td>[previously, pan, mi6, want, answers, take, li...</td>
      <td>[(previously, pan), (pan, mi6), (mi6, want), (...</td>
      <td>[(previously, pan, mi6), (pan, mi6, want), (mi...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pan.Am.S01E10.srt</td>
      <td>Previously on "Pan Am"...\n I bet you've got ...</td>
      <td>[previously, on, pan, am, i, bet, youve, got, ...</td>
      <td>[previously, pan, bet, youve, got, surprises, ...</td>
      <td>[(previously, pan), (pan, bet), (bet, youve), ...</td>
      <td>[(previously, pan, bet), (pan, bet, youve), (b...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Pan.Am.S01E04.srt</td>
      <td>Previously on "Pan Am"...\n - You're gonna me...</td>
      <td>[previously, on, pan, am, youre, gonna, meet, ...</td>
      <td>[previously, pan, youre, gonna, meet, kennedy,...</td>
      <td>[(previously, pan), (pan, youre), (youre, gonn...</td>
      <td>[(previously, pan, youre), (pan, youre, gonna)...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Pan.Am.S01E12.srt</td>
      <td>Previously on "Pan Am".\n We'd like to move y...</td>
      <td>[previously, on, pan, am, wed, like, to, move,...</td>
      <td>[previously, pan, wed, like, move, courier, ag...</td>
      <td>[(previously, pan), (pan, wed), (wed, like), (...</td>
      <td>[(previously, pan, wed), (pan, wed, like), (we...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Pan.Am.S01E06.srt</td>
      <td>Previously on "Pan Am".\n Why don't you came ...</td>
      <td>[previously, on, pan, am, why, dont, you, came...</td>
      <td>[previously, pan, dont, came, fog, captain, af...</td>
      <td>[(previously, pan), (pan, dont), (dont, came),...</td>
      <td>[(previously, pan, dont), (pan, dont, came), (...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Pan.Am.S01E07.srt</td>
      <td>Previously on "Pan Am"...\n You smell like wh...</td>
      <td>[previously, on, pan, am, you, smell, like, wh...</td>
      <td>[previously, pan, smell, like, whiskey, cigare...</td>
      <td>[(previously, pan), (pan, smell), (smell, like...</td>
      <td>[(previously, pan, smell), (pan, smell, like),...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Pan.Am.S01E13.srt</td>
      <td>Previously on "Pan Am"...\n Let's keep it in ...</td>
      <td>[previously, on, pan, am, lets, keep, it, in, ...</td>
      <td>[previously, pan, lets, keep, new, york, ginny...</td>
      <td>[(previously, pan), (pan, lets), (lets, keep),...</td>
      <td>[(previously, pan, lets), (pan, lets, keep), (...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Pan.Am.S01E03.srt</td>
      <td>Previously on "Pan Am".\n You're always disap...</td>
      <td>[previously, on, pan, am, youre, always, disap...</td>
      <td>[previously, pan, youre, always, disappearing,...</td>
      <td>[(previously, pan), (pan, youre), (youre, alwa...</td>
      <td>[(previously, pan, youre), (pan, youre, always...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pan.Am.S01E02.srt</td>
      <td>Previously on "Pan Am"...\n Do you not wanna ...</td>
      <td>[previously, on, pan, am, do, you, not, wanna,...</td>
      <td>[previously, pan, wanna, get, married, need, d...</td>
      <td>[(previously, pan), (pan, wanna), (wanna, get)...</td>
      <td>[(previously, pan, wanna), (pan, wanna, get), ...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Pan.Am.S01E14.srt</td>
      <td>Previously on "Pan Am"...\n There's a dealer ...</td>
      <td>[previously, on, pan, am, theres, a, dealer, i...</td>
      <td>[previously, pan, theres, dealer, london, whos...</td>
      <td>[(previously, pan), (pan, theres), (theres, de...</td>
      <td>[(previously, pan, theres), (pan, theres, deal...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Pan.Am.S01E01.srt</td>
      <td>ï»¿1\n There you are.\n Enjoy your flight.\n ...</td>
      <td>[ï»¿1, there, you, are, enjoy, your, flight, j...</td>
      <td>[ï»¿1, enjoy, flight, jet, clipper, service, u...</td>
      <td>[(ï»¿1, enjoy), (enjoy, flight), (flight, jet)...</td>
      <td>[(ï»¿1, enjoy, flight), (enjoy, flight, jet), ...</td>
    </tr>
  </tbody>
</table>
</div>


Putting it all together, the functions look like this:


```python
def convert_dict_df(script_dict):
    return pd.DataFrame.from_dict(script_dict, orient='index').reset_index().rename(columns={'index':'script_name', 0:'corpus'})

def punct_tokens(df, text_col):
    newline_list = '\t\r\n'
    remove_newline = str.maketrans(' ', ' ', newline_list)
    punct_list = string.punctuation
    nopunct = str.maketrans('', '', punct_list)
    df['no_punct_tokens'] = df[text_col].fillna("").str.lower().str.translate(remove_newline).str.translate(nopunct).str.split()
    return df

def create_ngrams(df):
    stop = nltk.corpus.stopwords.words('english')
    df['unigrams'] = df['no_punct_tokens'].apply(lambda x: [item for item in x if item not in stop])
    df['bigrams'] = df['unigrams'].apply(lambda x:(list(nltk.bigrams(x))))
    df['trigrams'] = df['unigrams'].apply(lambda x:(list(nltk.trigrams(x))))
    return df

def create_ngram_df(script_dict, text_col):
    df = convert_dict_df(script_dict)
    df = punct_tokens(df, text_col)
    df = create_ngrams(df)
    return df
```