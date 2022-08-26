---
title: "Entry NLP4: Frequencies and Comparison"
categories:
  - Blog
tags:
  - nlp
  - text analysis
---
# 

In the previous entries in this series, I loaded all the files in a directory, processed the data, and transformed it into ngrams. Now it's time to do math and analysis!


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

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/julie.fisher/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
# Grab and store the data
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
def convert_dict_df(script_dict):
    return pd.DataFrame.from_dict(script_dict, orient='index').reset_index().rename(columns={'index':'script_name', 0:'corpus'})

# Clean the text and create ngrams
def punct_tokens(df, text_col):
    newline_list = '\t\r\n'
    remove_newline = str.maketrans(' ', ' ', newline_list)
    punct_list = string.punctuation + '-‘_”'
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

# Frequencies

Counting words is a common sample problem and can probably be considered the 'hello world' of NLP. When putting it into a dictionary data structure, the concept isn't difficult:

- For each word (or in our case, n-gram) in the corpus
- Insert the word if it's not there (the dictionary key)
- Add 1 to the count (the dictionary value)

```
frequency_dictionary = {}
for ngram in ngram_list:
    if ngram not in frequency_dictionary:
        frequency_dictionary[ngram] = 0
    frequency_dictionary[ngram] +=1
```

The question is, how to apply this general concept to my specific use case.

The n-grams have already been created, so I don't have to worry about longer n-grams (the bigrams, and I threw in trigrams because why not?) spilling from one scrip to another. Which means I can concatenate all the n-grams of a specific category together (i.e. I don't want to combine unigrams with bigrams, just all the unigrams with each other).


```python
auth_file_path = os.path.join(os.getcwd(), 'data', '1960s')
raw_auth_dict = load_files_to_dict(auth_file_path, {})

auth_ngram_df = create_ngram_df(raw_auth_dict, 'corpus')
auth_ngram_df.head()
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



I already know I want to use the n-grams as my unique identifier, which means I'll need to create a separate dataframe for each set of frequencies - mixing unigrams with bigrams wouldn't let me do the analysis I want. This both simplifies and complicates the process, since I won't be able to just add on to the same dataframe anymore.

The `frequency_ct` and `dict_to_df` functions that I created in the previous solution to the homework still work. The only new aspect is that I need to put all the n-gram lists from the different scripts together. My initial thought was to use `list.expand`, but that would require looping through every row of the dataframe, which isn't the fastest or memory optimized solution.

Fortunately, there is an easy alternative: it's easily accomplished by using the `sum` method on the column as specified in this [StackOverflow answer](https://stackoverflow.com/a/42909969).


```python
auth_ngram_df['unigrams'].sum()[:10]
```




    ['youre',
     'traveling',
     'another',
     'dimension',
     'dimension',
     'sight',
     'sound',
     'mind',
     'journey',
     'wondrous']



Now that all of the ngrams are in a single list, it's a simple matter of creating a function to process them.


```python
def frequency_ct(ngram_list):
    freq_dict = {}
    for ngram in ngram_list:
        if ngram not in freq_dict:
            freq_dict[ngram] = 0
        freq_dict[ngram] +=1
    return freq_dict
```


```python
test_freq = frequency_ct(auth_ngram_df['unigrams'].sum())
test_freq
```




    {'youre': 1410,
     'traveling': 71,
     'another': 358,
     'dimension': 353,
     'sight': 131,
     'sound': 205,
     'mind': 422,
     'journey': 76,
     'wondrous': 72,
     'land': 180,
     'whose': 100,
     'boundaries': 60,
     'imagination': 138,
     'next': 390,
     'stop': 260,
     'twilight': 499,
     'zone': 506,
     'shes': 220,
     'set': 95,
     'mr': 1604,
     'radin': 27,
     'system': 36,
     'check': 118,
     'ready': 94,
     'go': 987,
     'dont': 2199,
     'know': 1777,
     'got': 1132,
     'effects': 8,
     'youd': 192,
     'swear': 29,
     'bomb': 52,
     'exploding': 3,
     'mean': 502,
     'big': 203,
     'thats': 1367,
     'precisely': 42,
     'way': 550,
     'supposed': 72,
     'quite': 171,
     'setup': 1,
     'part': 108,
     'illusion': 27,
     'room': 201,
     'venture': 3,
     'guess': 111,
     'best': 152,
     'designed': 23,
     'shelter': 22,
     'face': 121,
     'earth': 194,
     'knows': 94,
     'hydrogen': 5,
     'tonight': 160,
     'gags': 2,
     'huh': 256,
     'something': 662,
     'sort': 67,
     'practical': 9,
     'joke': 24,
     'lets': 338,
     'say': 657,
     'start': 115,
     'stuff': 71,
     'screen': 12,
     'world': 244,
     'getting': 137,
     'blasted': 2,
     'idea': 124,
     'three': 222,
     'guests': 8,
     'coming': 183,
     'evening': 66,
     'rather': 81,
     'special': 76,
     'fool': 30,
     'friends': 93,
     'must': 320,
     'kind': 307,
     'indeed': 93,
     'looked': 57,
     'takes': 62,
     'place': 279,
     '300': 19,
     'feet': 54,
     'underground': 6,
     'beneath': 5,
     'basement': 26,
     'new': 269,
     'york': 52,
     'city': 76,
     'skyscraper': 1,
     'owned': 6,
     'lived': 51,
     'one': 1304,
     'paul': 43,
     'rich': 27,
     'eccentric': 11,
     'singleminded': 2,
     'already': 107,
     'perceive': 6,
     'shall': 161,
     'see': 954,
     'moment': 181,
     'entered': 7,
     'good': 788,
     'step': 55,
     'across': 57,
     'hall': 25,
     'door': 197,
     'straight': 30,
     'ahead': 146,
     'please': 595,
     'come': 1071,
     'sit': 182,
     'make': 533,
     'comfortable': 21,
     'colonel': 92,
     'hawthorne': 1,
     'hughes': 4,
     'mrs': 195,
     'langsford': 16,
     'isnt': 387,
     'arent': 97,
     'excellent': 21,
     'memory': 23,
     'reverend': 19,
     'recognize': 22,
     'believe': 225,
     'served': 9,
     'didnt': 485,
     'second': 67,
     'lieutenant': 64,
     'infantry': 4,
     'regiment': 2,
     'command': 20,
     'africa': 8,
     '1942': 5,
     'recall': 28,
     'vaguely': 2,
     'seem': 55,
     'else': 196,
     'surprising': 2,
     'doesnt': 236,
     'flood': 2,
     'back': 733,
     'thousand': 29,
     'men': 207,
     'courtmartial': 2,
     'distinction': 9,
     'reserved': 3,
     'ah': 98,
     'yes': 907,
     'refused': 8,
     'lead': 28,
     'assault': 3,
     'hill': 10,
     'direct': 18,
     'order': 57,
     'delay': 3,
     'cost': 32,
     'us': 664,
     'almost': 79,
     'company': 52,
     'contention': 1,
     'board': 37,
     'stripped': 2,
     'rank': 8,
     'dishonorably': 1,
     'discharged': 9,
     'fortunate': 6,
     'dictated': 1,
     'sentence': 12,
     'would': 653,
     'shot': 79,
     'im': 1988,
     'sure': 449,
     'wretched': 1,
     'host': 2,
     'neglect': 1,
     'lady': 88,
     'present': 50,
     'course': 244,
     'taught': 8,
     'high': 54,
     'school': 53,
     'forget': 103,
     'students': 4,
     'oh': 1580,
     'sometimes': 49,
     'names': 39,
     'faces': 13,
     'get': 1286,
     'confused': 3,
     'prod': 2,
     'usually': 19,
     'connect': 5,
     'name': 267,
     'case': 107,
     'character': 23,
     'flunked': 4,
     'dressed': 11,
     'entire': 46,
     'class': 18,
     'called': 168,
     'humiliated': 4,
     'well': 2272,
     'let': 459,
     'delighted': 11,
     'accepted': 4,
     'invitation': 7,
     'request': 19,
     'ultimatum': 1,
     'chauffeur': 5,
     'said': 371,
     'matter': 286,
     'life': 274,
     'death': 131,
     'broached': 1,
     'dinner': 67,
     'wife': 135,
     'mary': 29,
     'went': 166,
     'answer': 81,
     'came': 186,
     'strange': 64,
     'expression': 5,
     'youve': 422,
     'never': 511,
     'ceased': 3,
     'bit': 95,
     'wordy': 1,
     'side': 55,
     'odd': 63,
     'really': 338,
     'changeless': 2,
     'remain': 45,
     'years': 344,
     'suppose': 92,
     'certain': 50,
     'lifetime': 6,
     'habits': 6,
     'easily': 13,
     'put': 300,
     'aside': 16,
     'perhaps': 111,
     'enough': 200,
     'tell': 894,
     'weve': 194,
     'asked': 70,
     'id': 303,
     'first': 283,
     'highball': 1,
     'cup': 36,
     'coffee': 73,
     'take': 641,
     'incredible': 47,
     'persistence': 1,
     'call': 313,
     'still': 240,
     'sitting': 44,
     'front': 61,
     'row': 11,
     'classroom': 2,
     'nice': 163,
     'tea': 21,
     'thank': 336,
     'tot': 1,
     'rum': 4,
     'appreciate': 32,
     'made': 222,
     'point': 114,
     'leave': 256,
     'obviously': 30,
     'welcome': 32,
     'hearing': 24,
     'whatever': 73,
     'staunchly': 1,
     'military': 10,
     'drive': 62,
     'objective': 1,
     'wipe': 9,
     'red': 30,
     'flag': 11,
     'map': 9,
     'troops': 4,
     'sun': 59,
     'nerves': 10,
     'steel': 37,
     'concrete': 8,
     'head': 118,
     'longer': 42,
     'attention': 32,
     'specific': 15,
     'purpose': 21,
     'want': 919,
     'settle': 18,
     'old': 526,
     'scores': 2,
     'chronology': 8,
     'dear': 166,
     'schoolmarm': 5,
     'begin': 34,
     'staunch': 2,
     'intrepid': 3,
     'educator': 1,
     'looks': 98,
     'without': 138,
     'severe': 4,
     'spectacles': 1,
     'covering': 7,
     'eyes': 86,
     'looking': 134,
     'possessing': 1,
     'vast': 11,
     'prerogative': 3,
     'comes': 110,
     'courage': 11,
     'pitting': 2,
     'wits': 3,
     'instinct': 2,
     'knowledge': 19,
     'captive': 1,
     'children': 77,
     'finished': 43,
     'ive': 627,
     'hardly': 31,
     'begun': 10,
     'may': 263,
     'observation': 13,
     'comment': 10,
     'whole': 173,
     'thing': 479,
     'man': 807,
     'like': 1309,
     'millionaire': 8,
     'times': 95,
     'important': 73,
     'walks': 12,
     'kings': 6,
     'heads': 32,
     'state': 87,
     'industrial': 4,
     'tycoons': 2,
     'tiny': 15,
     'could': 574,
     'brood': 2,
     'incident': 8,
     '20': 75,
     'ago': 162,
     'fester': 1,
     'inside': 91,
     'done': 177,
     'liked': 13,
     'humiliation': 4,
     'whether': 53,
     'happened': 170,
     'past': 67,
     'ten': 153,
     'minutes': 131,
     'right': 1428,
     'talk': 267,
     'caught': 34,
     'cheating': 6,
     'examination': 4,
     'crime': 13,
     'indicative': 1,
     'person': 51,
     'accused': 4,
     'act': 50,
     'cocoon': 1,
     'soon': 95,
     'become': 45,
     'tycoon': 3,
     'tried': 82,
     'plant': 26,
     'crib': 3,
     'sheets': 1,
     'innocent': 10,
     'student': 4,
     'stood': 14,
     'told': 272,
     'exactly': 102,
     'compassion': 16,
     'iota': 1,
     'sympathy': 11,
     'poor': 72,
     'frightened': 46,
     'desperate': 10,
     'boy': 311,
     'dealt': 1,
     'surprise': 35,
     'lent': 2,
     'neither': 30,
     'handed': 6,
     'wholesale': 6,
     'cheap': 27,
     'bubble': 3,
     'gum': 5,
     'recipient': 2,
     'worthy': 7,
     'devious': 2,
     'dishonest': 4,
     'troublemaker': 4,
     'spite': 5,
     'millions': 12,
     'doubt': 53,
     'even': 319,
     'havent': 184,
     'changed': 38,
     'many': 156,
     'passed': 13,
     'time': 817,
     'felt': 32,
     'suffered': 11,
     'indignities': 1,
     'hands': 82,
     'whats': 579,
     'gained': 6,
     'great': 165,
     'deal': 68,
     'example': 19,
     'lack': 11,
     'scandal': 2,
     'destroyed': 14,
     'reputation': 5,
     'remember': 247,
     'girl': 171,
     'drove': 14,
     'suicide': 16,
     'early': 40,
     'stage': 18,
     'held': 18,
     'honor': 26,
     'regard': 3,
     'devil': 49,
     'figure': 73,
     'speech': 18,
     'built': 51,
     'walls': 12,
     '18': 6,
     'inches': 4,
     'reinforced': 1,
     'around': 300,
     'six': 108,
     'generator': 2,
     'air': 74,
     'beyond': 95,
     'storeroom': 1,
     'size': 28,
     'warehouse': 2,
     'understand': 285,
     'logistics': 2,
     'occur': 16,
     'toyouwhy': 1,
     'gone': 119,
     'trouble': 96,
     'expense': 6,
     'vigil': 1,
     'long': 293,
     'wait': 290,
     'countdown': 5,
     'walked': 21,
     'listened': 9,
     'keep': 275,
     'abreast': 1,
     'things': 330,
     'going': 1139,
     'happen': 162,
     'pay': 114,
     'service': 23,
     'received': 10,
     'interesting': 32,
     'news': 40,
     'end': 101,
     'ladies': 49,
     'gentlemen': 106,
     '1145': 1,
     'country': 78,
     '30': 43,
     'midnight': 24,
     'dawn': 9,
     'nothing': 395,
     'left': 192,
     'rubble': 7,
     'bodies': 25,
     'moments': 25,
     'youll': 335,
     'sirens': 1,
     'shortly': 23,
     'alert': 8,
     'means': 91,
     'missiles': 7,
     'follow': 33,
     'survive': 27,
     'wish': 143,
     'rest': 130,
     'pallbearer': 3,
     'civil': 11,
     'defense': 16,
     'announcer': 5,
     'repeating': 5,
     'declared': 5,
     'takecover': 3,
     'signal': 14,
     'practice': 6,
     'warning': 12,
     'drill': 8,
     'attack': 34,
     'enemy': 30,
     'forces': 8,
     'expected': 10,
     'seek': 15,
     'nearest': 6,
     'immediately': 30,
     'home': 267,
     'prepared': 19,
     'toward': 23,
     'center': 12,
     'house': 134,
     'possible': 73,
     'outside': 81,
     'radio': 74,
     'type': 18,
     'building': 51,
     'lowest': 2,
     'floor': 55,
     'close': 79,
     'safe': 30,
     'test': 18,
     'real': 211,
     'comments': 4,
     'little': 615,
     'sophistry': 2,
     'quote': 4,
     'general': 94,
     'grant': 28,
     'enriching': 1,
     'gospel': 2,
     'silence': 9,
     'repertoire': 1,
     'pilgrims': 1,
     'progress': 21,
     'handle': 30,
     'situation': 19,
     'mental': 17,
     'eraser': 1,
     'reality': 13,
     'hold': 136,
     'die': 119,
     'together': 125,
     'turn': 83,
     'stomach': 19,
     'colonels': 2,
     'schoolmarms': 2,
     'precious': 16,
     'hide': 18,
     'sanctified': 1,
     'flesh': 26,
     'preoccupies': 1,
     'someone': 73,
     'love': 293,
     'somebody': 150,
     'theatrical': 2,
     'burlesque': 5,
     'legitimate': 5,
     'decency': 2,
     'depart': 7,
     'fragment': 6,
     'truth': 89,
     'mouth': 39,
     'scared': 53,
     'miserably': 1,
     'sell': 52,
     'pound': 6,
     'meant': 29,
     'survival': 16,
     'last': 282,
     'words': 45,
     'spoke': 12,
     'died': 61,
     'also': 75,
     'worst': 25,
     'falsehood': 2,
     'ever': 285,
     'uttered': 1,
     'open': 121,
     'possibly': 25,
     'known': 73,
     'difference': 62,
     'hell': 100,
     'reach': 42,
     'homes': 10,
     'happens': 87,
     'drop': 21,
     'pretenses': 2,
     'constructed': 1,
     'nonsense': 22,
     'walk': 109,
     'simply': 59,
     'live': 180,
     'permit': 8,
     'luxury': 7,
     'allow': 20,
     'stay': 198,
     'fact': 155,
     'destroy': 37,
     'ill': 896,
     'repay': 2,
     'compliment': 3,
     'require': 11,
     'eye': 41,
     'primitive': 5,
     'naked': 5,
     'price': 31,
     'interested': 22,
     'presume': 12,
     'submit': 5,
     'probably': 89,
     'gods': 11,
     'meaning': 11,
     'beg': 29,
     'pardon': 35,
     'ask': 171,
     'forgiveness': 6,
     'need': 237,
     'knees': 6,
     'perform': 9,
     'function': 21,
     'pretty': 110,
     'sugar': 19,
     'hows': 60,
     'speak': 76,
     'teacher': 5,
     'exact': 7,
     'favor': 30,
     'spend': 25,
     'quarter': 26,
     'hour': 91,
     'stray': 3,
     'cat': 24,
     'alone': 135,
     'central': 17,
     'park': 26,
     'full': 74,
     'strangers': 9,
     'blind': 14,
     'stupid': 41,
     'none': 57,
     'literally': 7,
     'string': 14,
     'silly': 30,
     'lesson': 12,
     'prayer': 11,
     'sorry': 258,
     'fine': 161,
     'five': 122,
     'theres': 509,
     'elevator': 23,
     'farce': 2,
     'conclusion': 8,
     'street': 101,
     'panic': 17,
     'frenzy': 1,
     'horror': 9,
     'salvation': 3,
     'watch': 97,
     'shoveled': 4,
     'grave': 21,
     'chance': 86,
     'stinking': 3,
     'throw': 41,
     'drain': 2,
     'infinitely': 2,
     'valuable': 10,
     'higher': 6,
     'expensive': 12,
     'amen': 152,
     'try': 199,
     'lonely': 29,
     'use': 96,
     'mirrors': 2,
     'help': 266,
     'radins': 1,
     'itll': 94,
     'fantasy': 19,
     'parade': 4,
     'illusions': 5,
     'people': 448,
     'justice': 17,
     'dignity': 8,
     'true': 97,
     'authorities': 11,
     'ordered': 11,
     'imminent': 4,
     'cover': 21,
     'car': 146,
     'driving': 19,
     'away': 305,
     'continue': 39,
     'movement': 7,
     'available': 14,
     'refuge': 2,
     'outdoors': 1,
     'foot': 26,
     'hurry': 56,
     'anybody': 101,
     'hey': 308,
     'mac': 36,
     'much': 405,
     'wont': 285,
     'listen': 241,
     'move': 128,
     'along': 87,
     'worry': 80,
     'okay': 175,
     'gonna': 108,
     'nobody': 101,
     'break': 63,
     'god': 137,
     'dealer': 9,
     'sits': 6,
     'making': 69,
     'imagines': 1,
     'hes': 529,
     'doomed': 7,
     'perdition': 1,
     'unutterable': 1,
     'loneliness': 17,
     'turned': 37,
     'nightmare': 34,
     'funeral': 16,
     'manufactured': 7,
     'rod': 59,
     'serling': 141,
     'creator': 56,
     'ofthe': 44,
     'weeks': 114,
     'story': 174,
     'message': 58,
     'week': 190,
     'offices': 6,
     'charles': 26,
     'beaumont': 13,
     'dead': 181,
     'mans': 46,
     'shoes': 20,
     'hobo': 2,
     'recentlydeceased': 1,
     'hoodlum': 1,
     'discovers': 4,
     'shoe': 6,
     'fits': 6,
     'wear': 36,
     'services': 24,
     'norm': 13,
     'hope': 114,
     'seat': 21,
     'belts': 7,
     'reduce': 3,
     'serious': 40,
     'injury': 3,
     'onethird': 3,
     'family': 48,
     'security': 9,
     'english': 21,
     'draw': 13,
     'perfect': 31,
     'position': 32,
     'fats': 48,
     'brown': 28,
     'sick': 90,
     'pool': 70,
     'cue': 13,
     'randolph': 14,
     'jesse': 40,
     'cardiff': 12,
     'hewasgood': 1,
     'hear': 297,
     '15': 40,
     'every': 235,
     'better': 323,
     'buried': 24,
     'ground': 34,
     'alive': 93,
     'beat': 65,
     'give': 370,
     'anything': 357,
     'play': 187,
     'game': 123,
     'report': 60,
     'listers': 2,
     'chicago': 10,
     'show': 150,
     'lousy': 23,
     'shark': 4,
     'learn': 42,
     'trying': 169,
     'carries': 11,
     'risks': 5,
     'um': 63,
     'legend': 6,
     'impossible': 38,
     'nothings': 12,
     'less': 41,
     'likely': 22,
     'others': 47,
     'rib': 3,
     'james': 40,
     'howard': 3,
     'shock': 20,
     'fire': 56,
     'cook': 30,
     'claim': 15,
     'deep': 33,
     'secondrate': 4,
     'minute': 217,
     'afraid': 144,
     'look': 658,
     'fooled': 7,
     'seen': 160,
     'skill': 5,
     'knack': 1,
     'style': 8,
     'heats': 1,
     'fold': 3,
     'fair': 58,
     'maybe': 326,
     'change': 100,
     'records': 6,
     'job': 130,
     'fat': 29,
     'tin': 9,
     'balloon': 7,
     'waiting': 117,
     'stick': 27,
     'needle': 8,
     'heres': 55,
     'legends': 5,
     'heard': 165,
     'saw': 149,
     'ninecushion': 1,
     'bank': 26,
     'hit': 59,
     'ball': 72,
     'hard': 60,
     'table': 31,
     'brains': 9,
     'yeah': 524,
     'stakes': 6,
     'money': 155,
     'worthwhile': 5,
     'talking': 180,
     'mister': 85,
     'lose': 54,
     'faith': 16,
     'bad': 120,
     'age': 61,
     'jumped': 10,
     'champion': 7,
     'equal': 10,
     'parts': 13,
     'talent': 39,
     'luck': 43,
     'work': 201,
     'nerve': 18,
     'quality': 7,
     'sadly': 1,
     'insanity': 7,
     'risk': 20,
     'prefer': 12,
     'player': 13,
     'mark': 18,
     'book': 67,
     'proud': 20,
     'wouldnt': 242,
     'wrong': 194,
     'hours': 153,
     'nights': 53,
     'slept': 5,
     'owner': 10,
     'closed': 31,
     'movies': 10,
     'dated': 2,
     'read': 98,
     'everything': 234,
     'pushing': 6,
     'race': 40,
     'driver': 15,
     'track': 21,
     'whisper': 6,
     'tazio': 1,
     'nuvolarl': 1,
     'nod': 3,
     'bullring': 1,
     'manolete': 1,
     'faced': 4,
     'daily': 4,
     'grade': 1,
     'playing': 40,
     'nutty': 5,
     'accept': 19,
     'terms': 14,
     'hunter': 5,
     'elephant': 1,
     'gun': 59,
     'fencer': 1,
     'uses': 5,
     'blade': 2,
     'lima': 1,
     'st': 5,
     'louis': 7,
     '600': 10,
     'yep': 6,
     'living': 88,
     '35': 5,
     'rotation': 1,
     'kelly': 9,
     'uh': 569,
     '141': 1,
     'rack': 5,
     'points': 12,
     'coin': 8,
     'toss': 2,
     'tails': 3,
     'thinking': 83,
     'son': 97,
     'breaks': 12,
     'disadvantage': 2,
     'scatters': 1,
     'balls': 7,
     'clear': 48,
     'field': 20,
     'wow': 7,
     'two': 386,
     'rail': 5,
     'advantage': 6,
     'given': 49,
     'scatter': 1,
     'funny': 81,
     'sink': 6,
     'run': 137,
     'miss': 389,
     'wide': 9,
     'nine': 26,
     'corner': 48,
     'cushion': 1,
     'kid': 83,
     'knew': 91,
     'lot': 136,
     'guys': 69,
     'music': 57,
     'basketball': 1,
     'feel': 209,
     'pocket': 20,
     'day': 306,
     '16': 7,
     'wandered': 3,
     'cool': 23,
     'dark': 50,
     'underwater': 3,
     'used': 163,
     'picked': 37,
     'played': 20,
     'geometry': 4,
     'mmhmm': 35,
     'challenging': 2,
     'form': 30,
     'science': 22,
     'precise': 6,
     'angles': 2,
     'sewed': 3,
     'ha': 17,
     'four': 136,
     'twelve': 4,
     '59': 2,
     'seven': 44,
     'young': 142,
     'yet': 115,
     'eleven': 6,
     'eight': 46,
     'needed': 25,
     'win': 39,
     'score': 10,
     '299': 1,
     '296': 1,
     'cooped': 6,
     'ought': 88,
     'bench': 1,
     'spent': 32,
     'took': 113,
     'places': 34,
     'billiards': 7,
     'uphill': 1,
     'swam': 1,
     'ocean': 16,
     'think': 868,
     'wonderful': 76,
     'hurts': 15,
     'rotting': 2,
     ...}



Of course now that I have my counts, I want to sort the n-grams from most frequent to least frequent. My favorite method to do this? DataFrames.

Unlike the previous `convert_dict_df` function, this one will need to be more flexible. It needs to be able to handle both the authentic 1960s corpus, all four of the modern corpora, and which ever n-grams I happen to be running. The addition of a couple of variables to handle column naming and a `sort_values` method takes care of it.

The `corpus_name` variable in particular is important later in the analysis. I'll need to compare the authentic corpus which was written in the 1960s about the 1960s to each of the corpora written in the 21st century about the 1960s. With the flow I've established, I'll need to merge dataframes to complete the analysis. This is most easily accomplished when the non-join-on columns have different names.

Example: If I join two dataframes with column names = `['unigram', 'frequency']` I'll end up with a single dataframe with the column names = `['unigram', 'x-frequency', 'y-frequency']`. I find these `x` and `y` prefixes less than informative and prefer to name my columns explicitly.


```python
def dict_to_df(freq_dict, gram_name, corpus_name):
    if (type(gram_name)==str) and (type(corpus_name)==str):
        pass
    else:
        print('gram_name and corpus_name variables must be strings')
    freq_colname = corpus_name+'_frequency'
    df = pd.DataFrame.from_dict(freq_dict, orient='index'
                               ).reset_index().rename(columns={'index':gram_name, 0:freq_colname}
                                                     ).sort_values(freq_colname, ascending=False)
    return df
```

But why stop my function at just the frequency? I also need normalized frequencies. Normalized frequencies level the playing field of straight counts when comparing corpora. With simple counts, a larger corpus will have n-grams with larger counts simply because there are more words overall than a smaller corpus. It doesn't necessarily reflect any relevant comparison.

Also, the homework problem requires getting ratios of the normalized frequencies later in the analysis.


```python
def normalized_freq(freq_df, corpus_name):
    freq_col_name = corpus_name + '_frequency'
    norm_col_name = corpus_name + '_norm_freq'
    total_ct = freq_df[freq_col_name].sum()
    freq_df[norm_col_name] = freq_df[freq_col_name]/total_ct
    return freq_df

def create_frequencies(ngram_list, gram_name, corpus_name):
    freq_dict = frequency_ct(ngram_list)
    freq_df = dict_to_df(freq_dict, gram_name, corpus_name)
    freq_df = normalized_freq(freq_df, corpus_name)
    return freq_df
```


```python
auth_freq_df = create_frequencies(auth_ngram_df['unigrams'].sum(), 'unigram', 'authentic')
auth_freq_df.head()
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
      <th>unigram</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>well</td>
      <td>2272</td>
      <td>0.012132</td>
    </tr>
    <tr>
      <th>25</th>
      <td>dont</td>
      <td>2199</td>
      <td>0.011742</td>
    </tr>
    <tr>
      <th>175</th>
      <td>im</td>
      <td>1988</td>
      <td>0.010616</td>
    </tr>
    <tr>
      <th>26</th>
      <td>know</td>
      <td>1777</td>
      <td>0.009489</td>
    </tr>
    <tr>
      <th>19</th>
      <td>mr</td>
      <td>1604</td>
      <td>0.008565</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_file_path = os.path.join(os.getcwd(), 'data', '21st-century')
raw_test_dict = load_files_to_dict(test_file_path, {})

test_ngram_dict = {}
for script_group in list(raw_test_dict.keys()):
    test_ngram_dict[script_group] = create_ngram_df(raw_test_dict[script_group], 'corpus')

test_freq_dict = {}
for script_group in list(test_ngram_dict.keys()):
    test_freq_dict[script_group] = create_frequencies(test_ngram_dict[script_group]['unigrams'].sum(), 'unigram', script_group)

test_freq_dict['Pan_Am'].head()
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
      <th>unigram</th>
      <th>Pan_Am_frequency</th>
      <th>Pan_Am_norm_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>67</th>
      <td>im</td>
      <td>489</td>
      <td>0.015189</td>
    </tr>
    <tr>
      <th>114</th>
      <td>oh</td>
      <td>407</td>
      <td>0.012642</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dont</td>
      <td>379</td>
      <td>0.011772</td>
    </tr>
    <tr>
      <th>50</th>
      <td>well</td>
      <td>373</td>
      <td>0.011586</td>
    </tr>
    <tr>
      <th>119</th>
      <td>know</td>
      <td>323</td>
      <td>0.010033</td>
    </tr>
  </tbody>
</table>
</div>



# Compare corpora



The last piece of this homework challenge is to compare the authentic corpus (wrtten regarding the 1960s and penned in the 1960s) to the four test corpora (written regarding the 1960s but not penned until the 21st century).

To compare anything to anything, first I need to combine the different dataframes holding my test corpora with the authentic corpus. I decided to do this by merging the values for the authentic data into each of the dataframes holding the values for the test data.


```python
compare_dict = {}
for script_group in list(test_freq_dict.keys()):
    compare_dict[script_group] = test_freq_dict[script_group].merge(auth_freq_df, on='unigram', how='outer').fillna(0)
```


```python
compare_dict['Pan_Am'].head()
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
      <th>unigram</th>
      <th>Pan_Am_frequency</th>
      <th>Pan_Am_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>im</td>
      <td>489.0</td>
      <td>0.015189</td>
      <td>1988.0</td>
      <td>0.010616</td>
    </tr>
    <tr>
      <th>1</th>
      <td>oh</td>
      <td>407.0</td>
      <td>0.012642</td>
      <td>1580.0</td>
      <td>0.008437</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dont</td>
      <td>379.0</td>
      <td>0.011772</td>
      <td>2199.0</td>
      <td>0.011742</td>
    </tr>
    <tr>
      <th>3</th>
      <td>well</td>
      <td>373.0</td>
      <td>0.011586</td>
      <td>2272.0</td>
      <td>0.012132</td>
    </tr>
    <tr>
      <th>4</th>
      <td>know</td>
      <td>323.0</td>
      <td>0.010033</td>
      <td>1777.0</td>
      <td>0.009489</td>
    </tr>
  </tbody>
</table>
</div>



The equation I implemented in the previous solution to this homework was:

```
df['norm_freq_ratio'] = df.loc[(df['imitation_norm_freq'] != 0
                               ) & (df['authentic_norm_freq'] != 0), 'imitation_norm_freq'
                              ]/df.loc[(df['imitation_norm_freq'] != 0
                                       ) & (df['authentic_norm_freq'] != 0), 'authentic_norm_freq']
```

In order to implement this in the various dataframes, I'll need a way to identify the appropriate columns, regardless of which dataframe I'm working with. This can be done by looking for 'norm_freq' in the column names - which will pull out the normalized frequency for both the authentic and test data.


```python
[compare_dict['Pan_Am'].columns[compare_dict['Pan_Am'].columns.str.contains('norm_freq')]]
```




    [Index(['Pan_Am_norm_freq', 'authentic_norm_freq'], dtype='object')]



Referencing the dataframe by the dictionary and script group name is getting rather tedious, so I can just set the dictionary/script name as the dataframe I'm working with. This has a much cleaner appearance and, more importantly, is easier to read. Regardless of how good (or not) code is, it's much more common to have to read code in order to improve, maintain, update, or repair it than write it. My philosophy is to make code as easy to read as possible, so that my future self can decipher what I was thinking when I wrote it the first time around.


```python
test = compare_dict['Pan_Am']
test_cols = test.columns[test.columns.str.contains('norm_freq')]
test_cols
```




    Index(['Pan_Am_norm_freq', 'authentic_norm_freq'], dtype='object')



Now I can update my code to the more readable version. Since I use the test dataframe as the left object and the authentic dataframe as the right object in the join, I can count on the fact that the test:authentic columns will always be in the same order.

As an added bonus, I only have to write to the dictionary once instead of the initial write, then the update with the new columns.


```python
compare_dict = {}
for script_group in list(test_freq_dict.keys()):
    df = test_freq_dict[script_group].merge(auth_freq_df, on='unigram', how='outer').fillna(0)
    freq_cols = df.columns[df.columns.str.contains('norm_freq')]
    df['norm_freq_ratio'] = df.loc[(df[freq_cols[0]]!=0) & (df[freq_cols[1]]!=0), freq_cols[0]] / df.loc[(df[freq_cols[0]]!=0) & (df[freq_cols[1]]!=0), freq_cols[1]]
    compare_dict[script_group] = df
```


```python
compare_dict['Pan_Am'].head()
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
      <th>unigram</th>
      <th>Pan_Am_frequency</th>
      <th>Pan_Am_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>im</td>
      <td>489.0</td>
      <td>0.015189</td>
      <td>1988.0</td>
      <td>0.010616</td>
      <td>1.430801</td>
    </tr>
    <tr>
      <th>1</th>
      <td>oh</td>
      <td>407.0</td>
      <td>0.012642</td>
      <td>1580.0</td>
      <td>0.008437</td>
      <td>1.498387</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dont</td>
      <td>379.0</td>
      <td>0.011772</td>
      <td>2199.0</td>
      <td>0.011742</td>
      <td>1.002538</td>
    </tr>
    <tr>
      <th>3</th>
      <td>well</td>
      <td>373.0</td>
      <td>0.011586</td>
      <td>2272.0</td>
      <td>0.012132</td>
      <td>0.954965</td>
    </tr>
    <tr>
      <th>4</th>
      <td>know</td>
      <td>323.0</td>
      <td>0.010033</td>
      <td>1777.0</td>
      <td>0.009489</td>
      <td>1.057309</td>
    </tr>
  </tbody>
</table>
</div>



## High Ratios

High ratios for the normalized frequency show unigrams that were used commonly in the 21st-century scripts, but were extremely rare (but present) in 1960s scripts.


```python
for script_group in compare_dict.keys():
    print(script_group)
    display(compare_dict[script_group].sort_values('norm_freq_ratio', ascending=False).head(50))
    print('\n')
```

    Pan_Am
    


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
      <th>unigram</th>
      <th>Pan_Am_frequency</th>
      <th>Pan_Am_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>dean</td>
      <td>87.0</td>
      <td>0.002702</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>506.064637</td>
    </tr>
    <tr>
      <th>18</th>
      <td>pan</td>
      <td>160.0</td>
      <td>0.004970</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>310.231195</td>
    </tr>
    <tr>
      <th>162</th>
      <td>amanda</td>
      <td>32.0</td>
      <td>0.000994</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>186.138717</td>
    </tr>
    <tr>
      <th>89</th>
      <td>stewardess</td>
      <td>54.0</td>
      <td>0.001677</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>157.054543</td>
    </tr>
    <tr>
      <th>197</th>
      <td>teddy</td>
      <td>27.0</td>
      <td>0.000839</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>157.054543</td>
    </tr>
    <tr>
      <th>281</th>
      <td>stewardesses</td>
      <td>19.0</td>
      <td>0.000590</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>110.519863</td>
    </tr>
    <tr>
      <th>364</th>
      <td>ryan</td>
      <td>15.0</td>
      <td>0.000466</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>87.252524</td>
    </tr>
    <tr>
      <th>456</th>
      <td>cia</td>
      <td>13.0</td>
      <td>0.000404</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>75.618854</td>
    </tr>
    <tr>
      <th>452</th>
      <td>ich</td>
      <td>13.0</td>
      <td>0.000404</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>75.618854</td>
    </tr>
    <tr>
      <th>491</th>
      <td>monte</td>
      <td>12.0</td>
      <td>0.000373</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>69.802019</td>
    </tr>
    <tr>
      <th>483</th>
      <td>omar</td>
      <td>12.0</td>
      <td>0.000373</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>69.802019</td>
    </tr>
    <tr>
      <th>84</th>
      <td>ii</td>
      <td>59.0</td>
      <td>0.001833</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>68.638652</td>
    </tr>
    <tr>
      <th>548</th>
      <td>carlo</td>
      <td>10.0</td>
      <td>0.000311</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>58.168349</td>
    </tr>
    <tr>
      <th>569</th>
      <td>monsieur</td>
      <td>10.0</td>
      <td>0.000311</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>58.168349</td>
    </tr>
    <tr>
      <th>37</th>
      <td>maggie</td>
      <td>108.0</td>
      <td>0.003355</td>
      <td>11.0</td>
      <td>0.000059</td>
      <td>57.110743</td>
    </tr>
    <tr>
      <th>603</th>
      <td>le</td>
      <td>9.0</td>
      <td>0.000280</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>52.351514</td>
    </tr>
    <tr>
      <th>608</th>
      <td>hier</td>
      <td>9.0</td>
      <td>0.000280</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>52.351514</td>
    </tr>
    <tr>
      <th>625</th>
      <td>soviets</td>
      <td>9.0</td>
      <td>0.000280</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>52.351514</td>
    </tr>
    <tr>
      <th>634</th>
      <td>lauras</td>
      <td>9.0</td>
      <td>0.000280</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>52.351514</td>
    </tr>
    <tr>
      <th>641</th>
      <td>courier</td>
      <td>9.0</td>
      <td>0.000280</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>52.351514</td>
    </tr>
    <tr>
      <th>713</th>
      <td>maggies</td>
      <td>8.0</td>
      <td>0.000248</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>46.534679</td>
    </tr>
    <tr>
      <th>349</th>
      <td>rio</td>
      <td>16.0</td>
      <td>0.000497</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>46.534679</td>
    </tr>
    <tr>
      <th>734</th>
      <td>moscow</td>
      <td>8.0</td>
      <td>0.000248</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>46.534679</td>
    </tr>
    <tr>
      <th>660</th>
      <td>zu</td>
      <td>8.0</td>
      <td>0.000248</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>46.534679</td>
    </tr>
    <tr>
      <th>81</th>
      <td>ted</td>
      <td>59.0</td>
      <td>0.001833</td>
      <td>8.0</td>
      <td>0.000043</td>
      <td>42.899157</td>
    </tr>
    <tr>
      <th>259</th>
      <td>greg</td>
      <td>21.0</td>
      <td>0.000652</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>40.717844</td>
    </tr>
    <tr>
      <th>764</th>
      <td>cockpit</td>
      <td>7.0</td>
      <td>0.000217</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>40.717844</td>
    </tr>
    <tr>
      <th>396</th>
      <td>magazine</td>
      <td>14.0</td>
      <td>0.000435</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>40.717844</td>
    </tr>
    <tr>
      <th>770</th>
      <td>casino</td>
      <td>7.0</td>
      <td>0.000217</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>40.717844</td>
    </tr>
    <tr>
      <th>818</th>
      <td>graham</td>
      <td>7.0</td>
      <td>0.000217</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>40.717844</td>
    </tr>
    <tr>
      <th>449</th>
      <td>previously</td>
      <td>13.0</td>
      <td>0.000404</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>37.809427</td>
    </tr>
    <tr>
      <th>925</th>
      <td>tasty</td>
      <td>6.0</td>
      <td>0.000186</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>34.901009</td>
    </tr>
    <tr>
      <th>855</th>
      <td>diplomatic</td>
      <td>6.0</td>
      <td>0.000186</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>34.901009</td>
    </tr>
    <tr>
      <th>883</th>
      <td>palace</td>
      <td>6.0</td>
      <td>0.000186</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>34.901009</td>
    </tr>
    <tr>
      <th>877</th>
      <td>cleared</td>
      <td>6.0</td>
      <td>0.000186</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>34.901009</td>
    </tr>
    <tr>
      <th>177</th>
      <td>rome</td>
      <td>30.0</td>
      <td>0.000932</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>34.901009</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>choosing</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>1041</th>
      <td>pudding</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>guessing</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>khrushchev</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>587</th>
      <td>runway</td>
      <td>10.0</td>
      <td>0.000311</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>987</th>
      <td>safely</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>988</th>
      <td>economy</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>993</th>
      <td>ugh</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.084175</td>
    </tr>
    <tr>
      <th>113</th>
      <td>london</td>
      <td>44.0</td>
      <td>0.001367</td>
      <td>9.0</td>
      <td>0.000048</td>
      <td>28.437860</td>
    </tr>
    <tr>
      <th>291</th>
      <td>anderson</td>
      <td>19.0</td>
      <td>0.000590</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>27.629966</td>
    </tr>
    <tr>
      <th>105</th>
      <td>mm</td>
      <td>46.0</td>
      <td>0.001429</td>
      <td>11.0</td>
      <td>0.000059</td>
      <td>24.324946</td>
    </tr>
    <tr>
      <th>468</th>
      <td>cargo</td>
      <td>12.0</td>
      <td>0.000373</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>23.267340</td>
    </tr>
    <tr>
      <th>1371</th>
      <td>fairly</td>
      <td>4.0</td>
      <td>0.000124</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>23.267340</td>
    </tr>
    <tr>
      <th>1336</th>
      <td>32</td>
      <td>4.0</td>
      <td>0.000124</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>23.267340</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Mad_Men
    


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
      <th>unigram</th>
      <th>Mad_Men_frequency</th>
      <th>Mad_Men_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>138</th>
      <td>sterling</td>
      <td>170.0</td>
      <td>0.001166</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>109.151410</td>
    </tr>
    <tr>
      <th>172</th>
      <td>sally</td>
      <td>143.0</td>
      <td>0.000981</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>91.815598</td>
    </tr>
    <tr>
      <th>54</th>
      <td>draper</td>
      <td>365.0</td>
      <td>0.002503</td>
      <td>6.0</td>
      <td>0.000032</td>
      <td>78.118166</td>
    </tr>
    <tr>
      <th>238</th>
      <td>jesus</td>
      <td>108.0</td>
      <td>0.000741</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>69.343249</td>
    </tr>
    <tr>
      <th>553</th>
      <td>francis</td>
      <td>42.0</td>
      <td>0.000288</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>53.933638</td>
    </tr>
    <tr>
      <th>317</th>
      <td>clients</td>
      <td>74.0</td>
      <td>0.000507</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>47.512967</td>
    </tr>
    <tr>
      <th>187</th>
      <td>joan</td>
      <td>134.0</td>
      <td>0.000919</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>43.018497</td>
    </tr>
    <tr>
      <th>195</th>
      <td>betty</td>
      <td>128.0</td>
      <td>0.000878</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>41.092295</td>
    </tr>
    <tr>
      <th>435</th>
      <td>jimmy</td>
      <td>55.0</td>
      <td>0.000377</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>35.313691</td>
    </tr>
    <tr>
      <th>457</th>
      <td>ken</td>
      <td>52.0</td>
      <td>0.000357</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>33.387490</td>
    </tr>
    <tr>
      <th>843</th>
      <td>crap</td>
      <td>26.0</td>
      <td>0.000178</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>33.387490</td>
    </tr>
    <tr>
      <th>931</th>
      <td>presentation</td>
      <td>23.0</td>
      <td>0.000158</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.535087</td>
    </tr>
    <tr>
      <th>905</th>
      <td>freddy</td>
      <td>23.0</td>
      <td>0.000158</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>29.535087</td>
    </tr>
    <tr>
      <th>354</th>
      <td>bobby</td>
      <td>67.0</td>
      <td>0.000459</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>28.678998</td>
    </tr>
    <tr>
      <th>358</th>
      <td>creative</td>
      <td>66.0</td>
      <td>0.000453</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>28.250953</td>
    </tr>
    <tr>
      <th>942</th>
      <td>holloway</td>
      <td>22.0</td>
      <td>0.000151</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>28.250953</td>
    </tr>
    <tr>
      <th>937</th>
      <td>clara</td>
      <td>22.0</td>
      <td>0.000151</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>28.250953</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>fatherinlaw</td>
      <td>20.0</td>
      <td>0.000137</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>25.682685</td>
    </tr>
    <tr>
      <th>994</th>
      <td>spectacular</td>
      <td>20.0</td>
      <td>0.000137</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>25.682685</td>
    </tr>
    <tr>
      <th>1035</th>
      <td>belle</td>
      <td>20.0</td>
      <td>0.000137</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>25.682685</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>joey</td>
      <td>19.0</td>
      <td>0.000130</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>24.398550</td>
    </tr>
    <tr>
      <th>1051</th>
      <td>whitman</td>
      <td>19.0</td>
      <td>0.000130</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>24.398550</td>
    </tr>
    <tr>
      <th>1050</th>
      <td>connie</td>
      <td>19.0</td>
      <td>0.000130</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>24.398550</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>delicious</td>
      <td>18.0</td>
      <td>0.000123</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>23.114416</td>
    </tr>
    <tr>
      <th>1120</th>
      <td>jewish</td>
      <td>18.0</td>
      <td>0.000123</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>23.114416</td>
    </tr>
    <tr>
      <th>451</th>
      <td>dick</td>
      <td>53.0</td>
      <td>0.000363</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>22.686371</td>
    </tr>
    <tr>
      <th>653</th>
      <td>airlines</td>
      <td>35.0</td>
      <td>0.000240</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>22.472349</td>
    </tr>
    <tr>
      <th>691</th>
      <td>partners</td>
      <td>33.0</td>
      <td>0.000226</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>21.188215</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>dallas</td>
      <td>16.0</td>
      <td>0.000110</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>20.546148</td>
    </tr>
    <tr>
      <th>1217</th>
      <td>strategy</td>
      <td>16.0</td>
      <td>0.000110</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>20.546148</td>
    </tr>
    <tr>
      <th>1216</th>
      <td>award</td>
      <td>16.0</td>
      <td>0.000110</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>20.546148</td>
    </tr>
    <tr>
      <th>1270</th>
      <td>reception</td>
      <td>15.0</td>
      <td>0.000103</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>19.262013</td>
    </tr>
    <tr>
      <th>1331</th>
      <td>danny</td>
      <td>15.0</td>
      <td>0.000103</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>19.262013</td>
    </tr>
    <tr>
      <th>1285</th>
      <td>episode</td>
      <td>15.0</td>
      <td>0.000103</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>19.262013</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>casting</td>
      <td>15.0</td>
      <td>0.000103</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>19.262013</td>
    </tr>
    <tr>
      <th>755</th>
      <td>previously</td>
      <td>29.0</td>
      <td>0.000199</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>18.619946</td>
    </tr>
    <tr>
      <th>1370</th>
      <td>suitcase</td>
      <td>14.0</td>
      <td>0.000096</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>17.977879</td>
    </tr>
    <tr>
      <th>1371</th>
      <td>cancel</td>
      <td>14.0</td>
      <td>0.000096</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>17.977879</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>grey</td>
      <td>14.0</td>
      <td>0.000096</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>17.977879</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>bowl</td>
      <td>13.0</td>
      <td>0.000089</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>16.693745</td>
    </tr>
    <tr>
      <th>364</th>
      <td>duck</td>
      <td>64.0</td>
      <td>0.000439</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>16.436918</td>
    </tr>
    <tr>
      <th>279</th>
      <td>lane</td>
      <td>89.0</td>
      <td>0.000610</td>
      <td>7.0</td>
      <td>0.000037</td>
      <td>16.326850</td>
    </tr>
    <tr>
      <th>635</th>
      <td>hare</td>
      <td>37.0</td>
      <td>0.000254</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>15.837656</td>
    </tr>
    <tr>
      <th>630</th>
      <td>beans</td>
      <td>37.0</td>
      <td>0.000254</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>15.837656</td>
    </tr>
    <tr>
      <th>636</th>
      <td>greg</td>
      <td>37.0</td>
      <td>0.000254</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>15.837656</td>
    </tr>
    <tr>
      <th>482</th>
      <td>campaign</td>
      <td>49.0</td>
      <td>0.000336</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>15.730644</td>
    </tr>
    <tr>
      <th>1536</th>
      <td>joyce</td>
      <td>12.0</td>
      <td>0.000082</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>15.409611</td>
    </tr>
    <tr>
      <th>1572</th>
      <td>chemical</td>
      <td>12.0</td>
      <td>0.000082</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>15.409611</td>
    </tr>
    <tr>
      <th>1584</th>
      <td>salad</td>
      <td>12.0</td>
      <td>0.000082</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>15.409611</td>
    </tr>
    <tr>
      <th>1600</th>
      <td>mens</td>
      <td>12.0</td>
      <td>0.000082</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>15.409611</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    X-Men_First_Class
    


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
      <th>unigram</th>
      <th>X-Men_First_Class_frequency</th>
      <th>X-Men_First_Class_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79</th>
      <td>cia</td>
      <td>10.0</td>
      <td>0.002281</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>427.076397</td>
    </tr>
    <tr>
      <th>171</th>
      <td>commands</td>
      <td>5.0</td>
      <td>0.001140</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>213.538198</td>
    </tr>
    <tr>
      <th>103</th>
      <td>cuba</td>
      <td>9.0</td>
      <td>0.002052</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>192.184379</td>
    </tr>
    <tr>
      <th>210</th>
      <td>sebastian</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>170.830559</td>
    </tr>
    <tr>
      <th>211</th>
      <td>shaws</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>170.830559</td>
    </tr>
    <tr>
      <th>364</th>
      <td>x</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>128.122919</td>
    </tr>
    <tr>
      <th>326</th>
      <td>presentation</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>128.122919</td>
    </tr>
    <tr>
      <th>284</th>
      <td>moscow</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>128.122919</td>
    </tr>
    <tr>
      <th>264</th>
      <td>threat</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>128.122919</td>
    </tr>
    <tr>
      <th>370</th>
      <td>homo</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>128.122919</td>
    </tr>
    <tr>
      <th>500</th>
      <td>jekyll</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>396</th>
      <td>groovy</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>540</th>
      <td>atom</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>417</th>
      <td>arrangement</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>511</th>
      <td>cola</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>434</th>
      <td>facility</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>577</th>
      <td>delicious</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>415</th>
      <td>florida</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>624</th>
      <td>formal</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>623</th>
      <td>dusseldorf</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>216</th>
      <td>argentina</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>595</th>
      <td>absorb</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>85.415279</td>
    </tr>
    <tr>
      <th>118</th>
      <td>russians</td>
      <td>7.0</td>
      <td>0.001596</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>74.738369</td>
    </tr>
    <tr>
      <th>115</th>
      <td>turkey</td>
      <td>8.0</td>
      <td>0.001824</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>68.332223</td>
    </tr>
    <tr>
      <th>116</th>
      <td>russia</td>
      <td>8.0</td>
      <td>0.001824</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>68.332223</td>
    </tr>
    <tr>
      <th>72</th>
      <td>missiles</td>
      <td>11.0</td>
      <td>0.002509</td>
      <td>7.0</td>
      <td>0.000037</td>
      <td>67.112005</td>
    </tr>
    <tr>
      <th>18</th>
      <td>hank</td>
      <td>23.0</td>
      <td>0.005245</td>
      <td>15.0</td>
      <td>0.000080</td>
      <td>65.485048</td>
    </tr>
    <tr>
      <th>335</th>
      <td>jesus</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>64.061460</td>
    </tr>
    <tr>
      <th>380</th>
      <td>usa</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>64.061460</td>
    </tr>
    <tr>
      <th>296</th>
      <td>reconsider</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>64.061460</td>
    </tr>
    <tr>
      <th>81</th>
      <td>wow</td>
      <td>10.0</td>
      <td>0.002281</td>
      <td>7.0</td>
      <td>0.000037</td>
      <td>61.010914</td>
    </tr>
    <tr>
      <th>230</th>
      <td>destination</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>56.943520</td>
    </tr>
    <tr>
      <th>164</th>
      <td>beast</td>
      <td>5.0</td>
      <td>0.001140</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>53.384550</td>
    </tr>
    <tr>
      <th>136</th>
      <td>soviet</td>
      <td>6.0</td>
      <td>0.001368</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>51.249168</td>
    </tr>
    <tr>
      <th>368</th>
      <td>rockets</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>13</th>
      <td>charles</td>
      <td>26.0</td>
      <td>0.005929</td>
      <td>26.0</td>
      <td>0.000139</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1117</th>
      <td>spectacular</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>oneway</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1088</th>
      <td>expectations</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>525</th>
      <td>backup</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>529</th>
      <td>nazis</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>532</th>
      <td>gates</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1124</th>
      <td>currently</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>575</th>
      <td>senior</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1065</th>
      <td>hoohoo</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>588</th>
      <td>freaks</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>573</th>
      <td>scratch</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>407</th>
      <td>serum</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>491</th>
      <td>mutated</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>42.707640</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>colleges</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>42.707640</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    The_Kennedys
    


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
      <th>unigram</th>
      <th>The_Kennedys_frequency</th>
      <th>The_Kennedys_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>bobby</td>
      <td>112.0</td>
      <td>0.006192</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>386.549750</td>
    </tr>
    <tr>
      <th>86</th>
      <td>khrushchev</td>
      <td>30.0</td>
      <td>0.001659</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>310.620335</td>
    </tr>
    <tr>
      <th>103</th>
      <td>sighs</td>
      <td>25.0</td>
      <td>0.001382</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>258.850279</td>
    </tr>
    <tr>
      <th>165</th>
      <td>rosemary</td>
      <td>18.0</td>
      <td>0.000995</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>186.372201</td>
    </tr>
    <tr>
      <th>12</th>
      <td>kennedy</td>
      <td>128.0</td>
      <td>0.007077</td>
      <td>9.0</td>
      <td>0.000048</td>
      <td>147.257048</td>
    </tr>
    <tr>
      <th>101</th>
      <td>cuba</td>
      <td>25.0</td>
      <td>0.001382</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>129.425140</td>
    </tr>
    <tr>
      <th>37</th>
      <td>ii</td>
      <td>60.0</td>
      <td>0.003317</td>
      <td>5.0</td>
      <td>0.000027</td>
      <td>124.248134</td>
    </tr>
    <tr>
      <th>298</th>
      <td>election</td>
      <td>11.0</td>
      <td>0.000608</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>113.894123</td>
    </tr>
    <tr>
      <th>163</th>
      <td>ethel</td>
      <td>18.0</td>
      <td>0.000995</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>93.186101</td>
    </tr>
    <tr>
      <th>399</th>
      <td>elected</td>
      <td>8.0</td>
      <td>0.000442</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>82.832089</td>
    </tr>
    <tr>
      <th>379</th>
      <td>dallas</td>
      <td>8.0</td>
      <td>0.000442</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>82.832089</td>
    </tr>
    <tr>
      <th>394</th>
      <td>mississippi</td>
      <td>8.0</td>
      <td>0.000442</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>82.832089</td>
    </tr>
    <tr>
      <th>418</th>
      <td>cabinet</td>
      <td>8.0</td>
      <td>0.000442</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>82.832089</td>
    </tr>
    <tr>
      <th>110</th>
      <td>senator</td>
      <td>23.0</td>
      <td>0.001272</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>79.380752</td>
    </tr>
    <tr>
      <th>460</th>
      <td>organized</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>72.478078</td>
    </tr>
    <tr>
      <th>214</th>
      <td>christ</td>
      <td>14.0</td>
      <td>0.000774</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>72.478078</td>
    </tr>
    <tr>
      <th>127</th>
      <td>meredith</td>
      <td>21.0</td>
      <td>0.001161</td>
      <td>3.0</td>
      <td>0.000016</td>
      <td>72.478078</td>
    </tr>
    <tr>
      <th>492</th>
      <td>cia</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>72.478078</td>
    </tr>
    <tr>
      <th>243</th>
      <td>lyndon</td>
      <td>13.0</td>
      <td>0.000719</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>67.301073</td>
    </tr>
    <tr>
      <th>583</th>
      <td>bastard</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>559</th>
      <td>bases</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>525</th>
      <td>francis</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>556</th>
      <td>option</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>590</th>
      <td>jolly</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>616</th>
      <td>dean</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>607</th>
      <td>mcnamara</td>
      <td>6.0</td>
      <td>0.000332</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>62.124067</td>
    </tr>
    <tr>
      <th>109</th>
      <td>campaign</td>
      <td>23.0</td>
      <td>0.001272</td>
      <td>4.0</td>
      <td>0.000021</td>
      <td>59.535564</td>
    </tr>
    <tr>
      <th>294</th>
      <td>campus</td>
      <td>11.0</td>
      <td>0.000608</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>56.947061</td>
    </tr>
    <tr>
      <th>725</th>
      <td>sites</td>
      <td>5.0</td>
      <td>0.000276</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>51.770056</td>
    </tr>
    <tr>
      <th>689</th>
      <td>bundy</td>
      <td>5.0</td>
      <td>0.000276</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>51.770056</td>
    </tr>
    <tr>
      <th>872</th>
      <td>regime</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>762</th>
      <td>threat</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>794</th>
      <td>perception</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>802</th>
      <td>largely</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>894</th>
      <td>disaster</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>775</th>
      <td>grunts</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>773</th>
      <td>handing</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>763</th>
      <td>humiliating</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>747</th>
      <td>itsits</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>836</th>
      <td>defeat</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>903</th>
      <td>diplomatic</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>740</th>
      <td>subs</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>857</th>
      <td>cancel</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>885</th>
      <td>grasp</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>798</th>
      <td>lodge</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>796</th>
      <td>operational</td>
      <td>4.0</td>
      <td>0.000221</td>
      <td>1.0</td>
      <td>0.000005</td>
      <td>41.416045</td>
    </tr>
    <tr>
      <th>467</th>
      <td>administration</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>36.239039</td>
    </tr>
    <tr>
      <th>439</th>
      <td>roosevelt</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>36.239039</td>
    </tr>
    <tr>
      <th>508</th>
      <td>interview</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>36.239039</td>
    </tr>
    <tr>
      <th>445</th>
      <td>jimmy</td>
      <td>7.0</td>
      <td>0.000387</td>
      <td>2.0</td>
      <td>0.000011</td>
      <td>36.239039</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    


```python
high_score_results = pd.DataFrame(columns = ['script', 'score'])
for script_group in compare_dict.keys():
    high_score_results = high_score_results.append(
        {'script':script_group,
         'score':compare_dict[script_group].sort_values('norm_freq_ratio', ascending=False).head(50)['norm_freq_ratio'].sum()
        }, ignore_index=True)
display(high_score_results.sort_values('score'))
print('Best performing corpus (lowest score) {}'.format(high_score_results.iloc[high_score_results['score'].idxmin(), 0]))
print('Worst performing corpus (highest score) {}'.format(high_score_results.iloc[high_score_results['score'].idxmax(), 0]))
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
      <th>script</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mad_Men</td>
      <td>1456.975643</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pan_Am</td>
      <td>3336.811190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The_Kennedys</td>
      <td>3980.829683</td>
    </tr>
    <tr>
      <th>2</th>
      <td>X-Men_First_Class</td>
      <td>4282.152672</td>
    </tr>
  </tbody>
</table>
</div>


    Best performing corpus (lowest score) Mad_Men
    Worst performing corpus (highest score) X-Men_First_Class
    

## Low Ratios

Low ratios for the normalized frequency show unigrams that were used commonly in 1960, but were rare in the 21st-century scripts.


```python
for script_group in compare_dict.keys():
    print(script_group)
    display(compare_dict[script_group].sort_values('norm_freq_ratio').head(50))
    print('\n')
```

    Pan_Am
    


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
      <th>unigram</th>
      <th>Pan_Am_frequency</th>
      <th>Pan_Am_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5337</th>
      <td>honey</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>152.0</td>
      <td>0.000812</td>
      <td>0.038269</td>
    </tr>
    <tr>
      <th>3797</th>
      <td>imagination</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>138.0</td>
      <td>0.000737</td>
      <td>0.042151</td>
    </tr>
    <tr>
      <th>3761</th>
      <td>ship</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>137.0</td>
      <td>0.000732</td>
      <td>0.042459</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>human</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>101.0</td>
      <td>0.000539</td>
      <td>0.057592</td>
    </tr>
    <tr>
      <th>3247</th>
      <td>major</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>76.0</td>
      <td>0.000406</td>
      <td>0.076537</td>
    </tr>
    <tr>
      <th>4218</th>
      <td>machine</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>75.0</td>
      <td>0.000400</td>
      <td>0.077558</td>
    </tr>
    <tr>
      <th>3352</th>
      <td>radio</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>74.0</td>
      <td>0.000395</td>
      <td>0.078606</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>jerry</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>69.0</td>
      <td>0.000368</td>
      <td>0.084302</td>
    </tr>
    <tr>
      <th>4146</th>
      <td>shadow</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>68.0</td>
      <td>0.000363</td>
      <td>0.085542</td>
    </tr>
    <tr>
      <th>3089</th>
      <td>martin</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>66.0</td>
      <td>0.000352</td>
      <td>0.088134</td>
    </tr>
    <tr>
      <th>3510</th>
      <td>jackie</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>65.0</td>
      <td>0.000347</td>
      <td>0.089490</td>
    </tr>
    <tr>
      <th>5794</th>
      <td>television</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>60.0</td>
      <td>0.000320</td>
      <td>0.096947</td>
    </tr>
    <tr>
      <th>5287</th>
      <td>gun</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>59.0</td>
      <td>0.000315</td>
      <td>0.098590</td>
    </tr>
    <tr>
      <th>4002</th>
      <td>floor</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>55.0</td>
      <td>0.000294</td>
      <td>0.105761</td>
    </tr>
    <tr>
      <th>3930</th>
      <td>aunt</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>54.0</td>
      <td>0.000288</td>
      <td>0.107719</td>
    </tr>
    <tr>
      <th>1227</th>
      <td>sound</td>
      <td>4.0</td>
      <td>0.000124</td>
      <td>205.0</td>
      <td>0.001095</td>
      <td>0.113499</td>
    </tr>
    <tr>
      <th>2584</th>
      <td>whose</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>100.0</td>
      <td>0.000534</td>
      <td>0.116337</td>
    </tr>
    <tr>
      <th>5824</th>
      <td>devil</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>49.0</td>
      <td>0.000262</td>
      <td>0.118711</td>
    </tr>
    <tr>
      <th>1297</th>
      <td>earth</td>
      <td>4.0</td>
      <td>0.000124</td>
      <td>194.0</td>
      <td>0.001036</td>
      <td>0.119935</td>
    </tr>
    <tr>
      <th>2324</th>
      <td>general</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>94.0</td>
      <td>0.000502</td>
      <td>0.123762</td>
    </tr>
    <tr>
      <th>2662</th>
      <td>alan</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>94.0</td>
      <td>0.000502</td>
      <td>0.123762</td>
    </tr>
    <tr>
      <th>4767</th>
      <td>mans</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>46.0</td>
      <td>0.000246</td>
      <td>0.126453</td>
    </tr>
    <tr>
      <th>3009</th>
      <td>rip</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>46.0</td>
      <td>0.000246</td>
      <td>0.126453</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>ought</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>88.0</td>
      <td>0.000470</td>
      <td>0.132201</td>
    </tr>
    <tr>
      <th>4626</th>
      <td>evil</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>44.0</td>
      <td>0.000235</td>
      <td>0.132201</td>
    </tr>
    <tr>
      <th>5683</th>
      <td>scene</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>44.0</td>
      <td>0.000235</td>
      <td>0.132201</td>
    </tr>
    <tr>
      <th>2469</th>
      <td>kids</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>85.0</td>
      <td>0.000454</td>
      <td>0.136867</td>
    </tr>
    <tr>
      <th>5001</th>
      <td>rid</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>42.0</td>
      <td>0.000224</td>
      <td>0.138496</td>
    </tr>
    <tr>
      <th>2146</th>
      <td>kid</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>83.0</td>
      <td>0.000443</td>
      <td>0.140165</td>
    </tr>
    <tr>
      <th>2976</th>
      <td>account</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>41.0</td>
      <td>0.000219</td>
      <td>0.141874</td>
    </tr>
    <tr>
      <th>3576</th>
      <td>peter</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>41.0</td>
      <td>0.000219</td>
      <td>0.141874</td>
    </tr>
    <tr>
      <th>2942</th>
      <td>agnes</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>39.0</td>
      <td>0.000208</td>
      <td>0.149150</td>
    </tr>
    <tr>
      <th>4801</th>
      <td>heaven</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>39.0</td>
      <td>0.000208</td>
      <td>0.149150</td>
    </tr>
    <tr>
      <th>4363</th>
      <td>destroy</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>37.0</td>
      <td>0.000198</td>
      <td>0.157212</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>steel</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>37.0</td>
      <td>0.000198</td>
      <td>0.157212</td>
    </tr>
    <tr>
      <th>2321</th>
      <td>dog</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>72.0</td>
      <td>0.000384</td>
      <td>0.161579</td>
    </tr>
    <tr>
      <th>4102</th>
      <td>eh</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>36.0</td>
      <td>0.000192</td>
      <td>0.161579</td>
    </tr>
    <tr>
      <th>1911</th>
      <td>ideas</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>71.0</td>
      <td>0.000379</td>
      <td>0.163855</td>
    </tr>
    <tr>
      <th>5318</th>
      <td>team</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>35.0</td>
      <td>0.000187</td>
      <td>0.166195</td>
    </tr>
    <tr>
      <th>3242</th>
      <td>broken</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>33.0</td>
      <td>0.000176</td>
      <td>0.176268</td>
    </tr>
    <tr>
      <th>3395</th>
      <td>100</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>33.0</td>
      <td>0.000176</td>
      <td>0.176268</td>
    </tr>
    <tr>
      <th>4813</th>
      <td>fellas</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>32.0</td>
      <td>0.000171</td>
      <td>0.181776</td>
    </tr>
    <tr>
      <th>4513</th>
      <td>harmon</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>32.0</td>
      <td>0.000171</td>
      <td>0.181776</td>
    </tr>
    <tr>
      <th>1692</th>
      <td>alive</td>
      <td>3.0</td>
      <td>0.000093</td>
      <td>93.0</td>
      <td>0.000497</td>
      <td>0.187640</td>
    </tr>
    <tr>
      <th>1658</th>
      <td>indeed</td>
      <td>3.0</td>
      <td>0.000093</td>
      <td>93.0</td>
      <td>0.000497</td>
      <td>0.187640</td>
    </tr>
    <tr>
      <th>1027</th>
      <td>town</td>
      <td>5.0</td>
      <td>0.000155</td>
      <td>155.0</td>
      <td>0.000828</td>
      <td>0.187640</td>
    </tr>
    <tr>
      <th>5646</th>
      <td>explanation</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>31.0</td>
      <td>0.000166</td>
      <td>0.187640</td>
    </tr>
    <tr>
      <th>3003</th>
      <td>fellow</td>
      <td>1.0</td>
      <td>0.000031</td>
      <td>31.0</td>
      <td>0.000166</td>
      <td>0.187640</td>
    </tr>
    <tr>
      <th>317</th>
      <td>old</td>
      <td>17.0</td>
      <td>0.000528</td>
      <td>526.0</td>
      <td>0.002809</td>
      <td>0.187997</td>
    </tr>
    <tr>
      <th>2147</th>
      <td>crossed</td>
      <td>2.0</td>
      <td>0.000062</td>
      <td>61.0</td>
      <td>0.000326</td>
      <td>0.190716</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    Mad_Men
    


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
      <th>unigram</th>
      <th>Mad_Men_frequency</th>
      <th>Mad_Men_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3985</th>
      <td>twilight</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>499.0</td>
      <td>0.002665</td>
      <td>0.007720</td>
    </tr>
    <tr>
      <th>3764</th>
      <td>zone</td>
      <td>4.0</td>
      <td>0.000027</td>
      <td>506.0</td>
      <td>0.002702</td>
      <td>0.010151</td>
    </tr>
    <tr>
      <th>12149</th>
      <td>doc</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>57.0</td>
      <td>0.000304</td>
      <td>0.022529</td>
    </tr>
    <tr>
      <th>3302</th>
      <td>captain</td>
      <td>4.0</td>
      <td>0.000027</td>
      <td>208.0</td>
      <td>0.001111</td>
      <td>0.024695</td>
    </tr>
    <tr>
      <th>10348</th>
      <td>commander</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>52.0</td>
      <td>0.000278</td>
      <td>0.024695</td>
    </tr>
    <tr>
      <th>9954</th>
      <td>emma</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>48.0</td>
      <td>0.000256</td>
      <td>0.026753</td>
    </tr>
    <tr>
      <th>10925</th>
      <td>ace</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>47.0</td>
      <td>0.000251</td>
      <td>0.027322</td>
    </tr>
    <tr>
      <th>10578</th>
      <td>schmidt</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>46.0</td>
      <td>0.000246</td>
      <td>0.027916</td>
    </tr>
    <tr>
      <th>7970</th>
      <td>base</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>45.0</td>
      <td>0.000240</td>
      <td>0.028536</td>
    </tr>
    <tr>
      <th>4642</th>
      <td>sight</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>131.0</td>
      <td>0.000700</td>
      <td>0.029408</td>
    </tr>
    <tr>
      <th>12626</th>
      <td>precisely</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>42.0</td>
      <td>0.000224</td>
      <td>0.030575</td>
    </tr>
    <tr>
      <th>7376</th>
      <td>destroy</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>37.0</td>
      <td>0.000198</td>
      <td>0.034706</td>
    </tr>
    <tr>
      <th>11629</th>
      <td>sergeant</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>37.0</td>
      <td>0.000198</td>
      <td>0.034706</td>
    </tr>
    <tr>
      <th>5882</th>
      <td>traveling</td>
      <td>2.0</td>
      <td>0.000014</td>
      <td>71.0</td>
      <td>0.000379</td>
      <td>0.036173</td>
    </tr>
    <tr>
      <th>7451</th>
      <td>access</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>34.0</td>
      <td>0.000182</td>
      <td>0.037769</td>
    </tr>
    <tr>
      <th>10221</th>
      <td>julius</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>33.0</td>
      <td>0.000176</td>
      <td>0.038913</td>
    </tr>
    <tr>
      <th>9446</th>
      <td>jess</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>33.0</td>
      <td>0.000176</td>
      <td>0.038913</td>
    </tr>
    <tr>
      <th>5213</th>
      <td>substance</td>
      <td>2.0</td>
      <td>0.000014</td>
      <td>63.0</td>
      <td>0.000336</td>
      <td>0.040766</td>
    </tr>
    <tr>
      <th>4142</th>
      <td>colonel</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>92.0</td>
      <td>0.000491</td>
      <td>0.041874</td>
    </tr>
    <tr>
      <th>10549</th>
      <td>radar</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>29.0</td>
      <td>0.000155</td>
      <td>0.044280</td>
    </tr>
    <tr>
      <th>3890</th>
      <td>magic</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>85.0</td>
      <td>0.000454</td>
      <td>0.045322</td>
    </tr>
    <tr>
      <th>3835</th>
      <td>mister</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>85.0</td>
      <td>0.000454</td>
      <td>0.045322</td>
    </tr>
    <tr>
      <th>9702</th>
      <td>alex</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>28.0</td>
      <td>0.000150</td>
      <td>0.045862</td>
    </tr>
    <tr>
      <th>9698</th>
      <td>grant</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>28.0</td>
      <td>0.000150</td>
      <td>0.045862</td>
    </tr>
    <tr>
      <th>7923</th>
      <td>driscoll</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>27.0</td>
      <td>0.000144</td>
      <td>0.047561</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>illusion</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>27.0</td>
      <td>0.000144</td>
      <td>0.047561</td>
    </tr>
    <tr>
      <th>11946</th>
      <td>witch</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>27.0</td>
      <td>0.000144</td>
      <td>0.047561</td>
    </tr>
    <tr>
      <th>8747</th>
      <td>reckon</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>26.0</td>
      <td>0.000139</td>
      <td>0.049390</td>
    </tr>
    <tr>
      <th>2545</th>
      <td>amen</td>
      <td>6.0</td>
      <td>0.000041</td>
      <td>152.0</td>
      <td>0.000812</td>
      <td>0.050690</td>
    </tr>
    <tr>
      <th>6240</th>
      <td>stations</td>
      <td>2.0</td>
      <td>0.000014</td>
      <td>49.0</td>
      <td>0.000262</td>
      <td>0.052414</td>
    </tr>
    <tr>
      <th>4150</th>
      <td>jerry</td>
      <td>3.0</td>
      <td>0.000021</td>
      <td>69.0</td>
      <td>0.000368</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>11773</th>
      <td>horn</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>23.0</td>
      <td>0.000123</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>12777</th>
      <td>christie</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>23.0</td>
      <td>0.000123</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>8236</th>
      <td>33</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>23.0</td>
      <td>0.000123</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>10139</th>
      <td>toward</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>23.0</td>
      <td>0.000123</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>12232</th>
      <td>shortly</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>23.0</td>
      <td>0.000123</td>
      <td>0.055832</td>
    </tr>
    <tr>
      <th>12410</th>
      <td>engines</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>22.0</td>
      <td>0.000117</td>
      <td>0.058370</td>
    </tr>
    <tr>
      <th>9830</th>
      <td>repeat</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>22.0</td>
      <td>0.000117</td>
      <td>0.058370</td>
    </tr>
    <tr>
      <th>8212</th>
      <td>barney</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>22.0</td>
      <td>0.000117</td>
      <td>0.058370</td>
    </tr>
    <tr>
      <th>7255</th>
      <td>main</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>21.0</td>
      <td>0.000112</td>
      <td>0.061149</td>
    </tr>
    <tr>
      <th>9793</th>
      <td>item</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>21.0</td>
      <td>0.000112</td>
      <td>0.061149</td>
    </tr>
    <tr>
      <th>11043</th>
      <td>function</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>21.0</td>
      <td>0.000112</td>
      <td>0.061149</td>
    </tr>
    <tr>
      <th>7471</th>
      <td>properly</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>21.0</td>
      <td>0.000112</td>
      <td>0.061149</td>
    </tr>
    <tr>
      <th>10641</th>
      <td>tonights</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>21.0</td>
      <td>0.000112</td>
      <td>0.061149</td>
    </tr>
    <tr>
      <th>10496</th>
      <td>jenny</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>19.0</td>
      <td>0.000101</td>
      <td>0.067586</td>
    </tr>
    <tr>
      <th>10444</th>
      <td>wings</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>19.0</td>
      <td>0.000101</td>
      <td>0.067586</td>
    </tr>
    <tr>
      <th>9436</th>
      <td>ross</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>19.0</td>
      <td>0.000101</td>
      <td>0.067586</td>
    </tr>
    <tr>
      <th>10725</th>
      <td>degree</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>19.0</td>
      <td>0.000101</td>
      <td>0.067586</td>
    </tr>
    <tr>
      <th>9437</th>
      <td>reverend</td>
      <td>1.0</td>
      <td>0.000007</td>
      <td>19.0</td>
      <td>0.000101</td>
      <td>0.067586</td>
    </tr>
    <tr>
      <th>6728</th>
      <td>doll</td>
      <td>2.0</td>
      <td>0.000014</td>
      <td>37.0</td>
      <td>0.000198</td>
      <td>0.069413</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    X-Men_First_Class
    


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
      <th>unigram</th>
      <th>X-Men_First_Class_frequency</th>
      <th>X-Men_First_Class_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>212</th>
      <td>mr</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>1604.0</td>
      <td>0.008565</td>
      <td>0.106503</td>
    </tr>
    <tr>
      <th>1516</th>
      <td>boy</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>311.0</td>
      <td>0.001661</td>
      <td>0.137324</td>
    </tr>
    <tr>
      <th>1532</th>
      <td>away</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>305.0</td>
      <td>0.001629</td>
      <td>0.140025</td>
    </tr>
    <tr>
      <th>874</th>
      <td>hear</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>297.0</td>
      <td>0.001586</td>
      <td>0.143797</td>
    </tr>
    <tr>
      <th>782</th>
      <td>long</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>293.0</td>
      <td>0.001565</td>
      <td>0.145760</td>
    </tr>
    <tr>
      <th>405</th>
      <td>old</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>526.0</td>
      <td>0.002809</td>
      <td>0.162386</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>minute</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>217.0</td>
      <td>0.001159</td>
      <td>0.196809</td>
    </tr>
    <tr>
      <th>986</th>
      <td>captain</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>208.0</td>
      <td>0.001111</td>
      <td>0.205325</td>
    </tr>
    <tr>
      <th>1374</th>
      <td>room</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>201.0</td>
      <td>0.001073</td>
      <td>0.212476</td>
    </tr>
    <tr>
      <th>479</th>
      <td>night</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>394.0</td>
      <td>0.002104</td>
      <td>0.216790</td>
    </tr>
    <tr>
      <th>1401</th>
      <td>mrs</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>195.0</td>
      <td>0.001041</td>
      <td>0.219014</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>doctor</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>182.0</td>
      <td>0.000972</td>
      <td>0.234657</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>sit</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>182.0</td>
      <td>0.000972</td>
      <td>0.234657</td>
    </tr>
    <tr>
      <th>677</th>
      <td>dead</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>181.0</td>
      <td>0.000967</td>
      <td>0.235954</td>
    </tr>
    <tr>
      <th>1500</th>
      <td>land</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>180.0</td>
      <td>0.000961</td>
      <td>0.237265</td>
    </tr>
    <tr>
      <th>94</th>
      <td>oh</td>
      <td>9.0</td>
      <td>0.002052</td>
      <td>1580.0</td>
      <td>0.008437</td>
      <td>0.243271</td>
    </tr>
    <tr>
      <th>673</th>
      <td>quite</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>171.0</td>
      <td>0.000913</td>
      <td>0.249752</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>girl</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>171.0</td>
      <td>0.000913</td>
      <td>0.249752</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>trying</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>169.0</td>
      <td>0.000902</td>
      <td>0.252708</td>
    </tr>
    <tr>
      <th>376</th>
      <td>mean</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>502.0</td>
      <td>0.002681</td>
      <td>0.255225</td>
    </tr>
    <tr>
      <th>635</th>
      <td>dear</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>166.0</td>
      <td>0.000886</td>
      <td>0.257275</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>heard</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>165.0</td>
      <td>0.000881</td>
      <td>0.258834</td>
    </tr>
    <tr>
      <th>859</th>
      <td>ago</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>162.0</td>
      <td>0.000865</td>
      <td>0.263627</td>
    </tr>
    <tr>
      <th>756</th>
      <td>fine</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>161.0</td>
      <td>0.000860</td>
      <td>0.265265</td>
    </tr>
    <tr>
      <th>1429</th>
      <td>tonight</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>160.0</td>
      <td>0.000854</td>
      <td>0.266923</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>fact</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>155.0</td>
      <td>0.000828</td>
      <td>0.275533</td>
    </tr>
    <tr>
      <th>1361</th>
      <td>town</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>155.0</td>
      <td>0.000828</td>
      <td>0.275533</td>
    </tr>
    <tr>
      <th>1326</th>
      <td>hours</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>153.0</td>
      <td>0.000817</td>
      <td>0.279135</td>
    </tr>
    <tr>
      <th>1363</th>
      <td>honey</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>152.0</td>
      <td>0.000812</td>
      <td>0.280971</td>
    </tr>
    <tr>
      <th>42</th>
      <td>well</td>
      <td>15.0</td>
      <td>0.003421</td>
      <td>2272.0</td>
      <td>0.012132</td>
      <td>0.281961</td>
    </tr>
    <tr>
      <th>406</th>
      <td>around</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>300.0</td>
      <td>0.001602</td>
      <td>0.284718</td>
    </tr>
    <tr>
      <th>1469</th>
      <td>car</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>146.0</td>
      <td>0.000780</td>
      <td>0.292518</td>
    </tr>
    <tr>
      <th>527</th>
      <td>understand</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>285.0</td>
      <td>0.001522</td>
      <td>0.299703</td>
    </tr>
    <tr>
      <th>261</th>
      <td>uh</td>
      <td>4.0</td>
      <td>0.000912</td>
      <td>569.0</td>
      <td>0.003038</td>
      <td>0.300229</td>
    </tr>
    <tr>
      <th>478</th>
      <td>last</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>282.0</td>
      <td>0.001506</td>
      <td>0.302891</td>
    </tr>
    <tr>
      <th>334</th>
      <td>youve</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>422.0</td>
      <td>0.002253</td>
      <td>0.303609</td>
    </tr>
    <tr>
      <th>1056</th>
      <td>father</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>139.0</td>
      <td>0.000742</td>
      <td>0.307249</td>
    </tr>
    <tr>
      <th>1197</th>
      <td>getting</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>137.0</td>
      <td>0.000732</td>
      <td>0.311735</td>
    </tr>
    <tr>
      <th>797</th>
      <td>four</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>136.0</td>
      <td>0.000726</td>
      <td>0.314027</td>
    </tr>
    <tr>
      <th>418</th>
      <td>told</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>272.0</td>
      <td>0.001452</td>
      <td>0.314027</td>
    </tr>
    <tr>
      <th>1314</th>
      <td>house</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>134.0</td>
      <td>0.000716</td>
      <td>0.318714</td>
    </tr>
    <tr>
      <th>766</th>
      <td>late</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>134.0</td>
      <td>0.000716</td>
      <td>0.318714</td>
    </tr>
    <tr>
      <th>360</th>
      <td>next</td>
      <td>3.0</td>
      <td>0.000684</td>
      <td>390.0</td>
      <td>0.002083</td>
      <td>0.328520</td>
    </tr>
    <tr>
      <th>558</th>
      <td>huh</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>256.0</td>
      <td>0.001367</td>
      <td>0.333653</td>
    </tr>
    <tr>
      <th>1240</th>
      <td>game</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>123.0</td>
      <td>0.000657</td>
      <td>0.347217</td>
    </tr>
    <tr>
      <th>791</th>
      <td>five</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>122.0</td>
      <td>0.000651</td>
      <td>0.350063</td>
    </tr>
    <tr>
      <th>381</th>
      <td>morning</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>243.0</td>
      <td>0.001298</td>
      <td>0.351503</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>check</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>118.0</td>
      <td>0.000630</td>
      <td>0.361929</td>
    </tr>
    <tr>
      <th>742</th>
      <td>says</td>
      <td>1.0</td>
      <td>0.000228</td>
      <td>117.0</td>
      <td>0.000625</td>
      <td>0.365023</td>
    </tr>
    <tr>
      <th>473</th>
      <td>everything</td>
      <td>2.0</td>
      <td>0.000456</td>
      <td>234.0</td>
      <td>0.001250</td>
      <td>0.365023</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    The_Kennedys
    


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
      <th>unigram</th>
      <th>The_Kennedys_frequency</th>
      <th>The_Kennedys_norm_freq</th>
      <th>authentic_frequency</th>
      <th>authentic_norm_freq</th>
      <th>norm_freq_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1774</th>
      <td>zone</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>506.0</td>
      <td>0.002702</td>
      <td>0.040925</td>
    </tr>
    <tr>
      <th>1874</th>
      <td>earth</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>194.0</td>
      <td>0.001036</td>
      <td>0.053371</td>
    </tr>
    <tr>
      <th>2646</th>
      <td>game</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>123.0</td>
      <td>0.000657</td>
      <td>0.084179</td>
    </tr>
    <tr>
      <th>3250</th>
      <td>guess</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>111.0</td>
      <td>0.000593</td>
      <td>0.093279</td>
    </tr>
    <tr>
      <th>3314</th>
      <td>kill</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>106.0</td>
      <td>0.000566</td>
      <td>0.097679</td>
    </tr>
    <tr>
      <th>1328</th>
      <td>sound</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>205.0</td>
      <td>0.001095</td>
      <td>0.101015</td>
    </tr>
    <tr>
      <th>2802</th>
      <td>hot</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>101.0</td>
      <td>0.000539</td>
      <td>0.102515</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>key</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>97.0</td>
      <td>0.000518</td>
      <td>0.106742</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>space</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>90.0</td>
      <td>0.000481</td>
      <td>0.115045</td>
    </tr>
    <tr>
      <th>2352</th>
      <td>ought</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>88.0</td>
      <td>0.000470</td>
      <td>0.117659</td>
    </tr>
    <tr>
      <th>1618</th>
      <td>story</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>174.0</td>
      <td>0.000929</td>
      <td>0.119012</td>
    </tr>
    <tr>
      <th>2584</th>
      <td>hate</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>81.0</td>
      <td>0.000433</td>
      <td>0.127827</td>
    </tr>
    <tr>
      <th>2791</th>
      <td>black</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>77.0</td>
      <td>0.000411</td>
      <td>0.134468</td>
    </tr>
    <tr>
      <th>1581</th>
      <td>ten</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>153.0</td>
      <td>0.000817</td>
      <td>0.135347</td>
    </tr>
    <tr>
      <th>1760</th>
      <td>amen</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>152.0</td>
      <td>0.000812</td>
      <td>0.136237</td>
    </tr>
    <tr>
      <th>2156</th>
      <td>cold</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>76.0</td>
      <td>0.000406</td>
      <td>0.136237</td>
    </tr>
    <tr>
      <th>3020</th>
      <td>darling</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>76.0</td>
      <td>0.000406</td>
      <td>0.136237</td>
    </tr>
    <tr>
      <th>2852</th>
      <td>stuff</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>71.0</td>
      <td>0.000379</td>
      <td>0.145831</td>
    </tr>
    <tr>
      <th>2310</th>
      <td>pool</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>70.0</td>
      <td>0.000374</td>
      <td>0.147914</td>
    </tr>
    <tr>
      <th>3095</th>
      <td>shoot</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>70.0</td>
      <td>0.000374</td>
      <td>0.147914</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>ship</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>137.0</td>
      <td>0.000732</td>
      <td>0.151153</td>
    </tr>
    <tr>
      <th>3414</th>
      <td>book</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>67.0</td>
      <td>0.000358</td>
      <td>0.154537</td>
    </tr>
    <tr>
      <th>3394</th>
      <td>martin</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>66.0</td>
      <td>0.000352</td>
      <td>0.156879</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>death</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>131.0</td>
      <td>0.000700</td>
      <td>0.158077</td>
    </tr>
    <tr>
      <th>2172</th>
      <td>odd</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>63.0</td>
      <td>0.000336</td>
      <td>0.164349</td>
    </tr>
    <tr>
      <th>928</th>
      <td>play</td>
      <td>3.0</td>
      <td>0.000166</td>
      <td>187.0</td>
      <td>0.000999</td>
      <td>0.166107</td>
    </tr>
    <tr>
      <th>2229</th>
      <td>wonder</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>61.0</td>
      <td>0.000326</td>
      <td>0.169738</td>
    </tr>
    <tr>
      <th>2543</th>
      <td>boundaries</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>60.0</td>
      <td>0.000320</td>
      <td>0.172567</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>land</td>
      <td>3.0</td>
      <td>0.000166</td>
      <td>180.0</td>
      <td>0.000961</td>
      <td>0.172567</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>hit</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>59.0</td>
      <td>0.000315</td>
      <td>0.175492</td>
    </tr>
    <tr>
      <th>3638</th>
      <td>message</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>58.0</td>
      <td>0.000310</td>
      <td>0.178517</td>
    </tr>
    <tr>
      <th>3135</th>
      <td>fair</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>58.0</td>
      <td>0.000310</td>
      <td>0.178517</td>
    </tr>
    <tr>
      <th>2976</th>
      <td>charlie</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>58.0</td>
      <td>0.000310</td>
      <td>0.178517</td>
    </tr>
    <tr>
      <th>2269</th>
      <td>across</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>57.0</td>
      <td>0.000304</td>
      <td>0.181649</td>
    </tr>
    <tr>
      <th>2478</th>
      <td>pick</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>56.0</td>
      <td>0.000299</td>
      <td>0.184893</td>
    </tr>
    <tr>
      <th>1869</th>
      <td>creator</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>56.0</td>
      <td>0.000299</td>
      <td>0.184893</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>floor</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>55.0</td>
      <td>0.000294</td>
      <td>0.188255</td>
    </tr>
    <tr>
      <th>3388</th>
      <td>heres</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>55.0</td>
      <td>0.000294</td>
      <td>0.188255</td>
    </tr>
    <tr>
      <th>1683</th>
      <td>case</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>107.0</td>
      <td>0.000571</td>
      <td>0.193533</td>
    </tr>
    <tr>
      <th>2696</th>
      <td>nights</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>53.0</td>
      <td>0.000283</td>
      <td>0.195359</td>
    </tr>
    <tr>
      <th>317</th>
      <td>old</td>
      <td>10.0</td>
      <td>0.000553</td>
      <td>526.0</td>
      <td>0.002809</td>
      <td>0.196844</td>
    </tr>
    <tr>
      <th>2938</th>
      <td>company</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>52.0</td>
      <td>0.000278</td>
      <td>0.199116</td>
    </tr>
    <tr>
      <th>3031</th>
      <td>somewhere</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>52.0</td>
      <td>0.000278</td>
      <td>0.199116</td>
    </tr>
    <tr>
      <th>2871</th>
      <td>apartment</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>52.0</td>
      <td>0.000278</td>
      <td>0.199116</td>
    </tr>
    <tr>
      <th>3685</th>
      <td>bomb</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>52.0</td>
      <td>0.000278</td>
      <td>0.199116</td>
    </tr>
    <tr>
      <th>2213</th>
      <td>station</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>51.0</td>
      <td>0.000272</td>
      <td>0.203020</td>
    </tr>
    <tr>
      <th>2656</th>
      <td>person</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>51.0</td>
      <td>0.000272</td>
      <td>0.203020</td>
    </tr>
    <tr>
      <th>1706</th>
      <td>street</td>
      <td>2.0</td>
      <td>0.000111</td>
      <td>101.0</td>
      <td>0.000539</td>
      <td>0.205030</td>
    </tr>
    <tr>
      <th>1875</th>
      <td>named</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>50.0</td>
      <td>0.000267</td>
      <td>0.207080</td>
    </tr>
    <tr>
      <th>3509</th>
      <td>dark</td>
      <td>1.0</td>
      <td>0.000055</td>
      <td>50.0</td>
      <td>0.000267</td>
      <td>0.207080</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    


```python
low_score_results = pd.DataFrame(columns = ['script', 'score'])
for script_group in compare_dict.keys():
    low_score_results = low_score_results.append(
        {'script':script_group,
         'score':compare_dict[script_group].sort_values('norm_freq_ratio').head(50)['norm_freq_ratio'].sum()
        }, ignore_index=True)
display(low_score_results.sort_values('score', ascending=False))
print('Best performing corpus (highest score) {}'.format(low_score_results.iloc[low_score_results['score'].idxmax(), 0]))
print('Worst performing corpus (lowest score) {}'.format(low_score_results.iloc[low_score_results['score'].idxmin(), 0]))
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
      <th>script</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>X-Men_First_Class</td>
      <td>13.255571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The_Kennedys</td>
      <td>7.791826</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pan_Am</td>
      <td>6.533376</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mad_Men</td>
      <td>2.309133</td>
    </tr>
  </tbody>
</table>
</div>


    Best performing corpus (highest score) X-Men_First_Class
    Worst performing corpus (lowest score) Mad_Men
    

# Ranking

The scores returned both as top and bottom normalized frequency ratios are bad things:

- The 50 highest ratios are words that were used frequently in the 21st century scripts, but were rare in the 1960s
- the 50 lowest ratios are words that were used frequently in the 1960s, but showed up rarely in the 21st century scripts

In the high ratios set, the higher the ratio, the further the script is from the authentic corpus. In the low ratios set, the higher the ratio, the closer the script is to the authentic corpus. So to get my ranking, I'm going to subtract the low ratio from the high ratio. The script corpora will then be sorted from lowest (best) to highest (worst) score.


```python
results = pd.DataFrame(columns = ['script', 'high_ratio', 'low_ratio'])
for script_group in compare_dict.keys():
    results = results.append(
        {'script':script_group,
         'high_ratio':compare_dict[script_group].sort_values('norm_freq_ratio', ascending=False).head(50)['norm_freq_ratio'].sum(),
         'low_ratio':compare_dict[script_group].sort_values('norm_freq_ratio').head(50)['norm_freq_ratio'].sum()
        }, ignore_index=True)
    results['combined_score'] = results['high_ratio'] - results['low_ratio']
    results = results.sort_values('combined_score')
    results['rank'] = range(1, 1+len(results))
display(results)
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
      <th>script</th>
      <th>high_ratio</th>
      <th>low_ratio</th>
      <th>combined_score</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mad_Men</td>
      <td>1456.975643</td>
      <td>2.309133</td>
      <td>1454.666510</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pan_Am</td>
      <td>3336.811190</td>
      <td>6.533376</td>
      <td>3330.277814</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The_Kennedys</td>
      <td>3980.829683</td>
      <td>7.791826</td>
      <td>3973.037857</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>X-Men_First_Class</td>
      <td>4282.152672</td>
      <td>13.255571</td>
      <td>4268.897101</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


The analysis for the unigrams is now complete. To see the clean code (including improvements to functions) and the results for unigrams, bigrams, and trigrams, see the accompanying notebook.

# Caveats

There are several problems with this exercise and the solution.

## Corpus data processing

The biggest initial problem for me was the fact that punctuation wasn't removed, the n-grams were case sensitive, and stopwords weren't removed. The first two mean that words aren't counted appropriately, especially when they're prone to different capitalizations and uses with punctuation. For example, in the initial solution I noticed 'daddy' written several ways. Here are several ways 'daddy' could be included in a script

- Daddy.
- daddy.
- Daddy
- daddy
- Daddy!
- daddy!

This is six iterations for a single word which should all be counded together.

The last point, stopwords weren't removed, means that there's a lot of meaningless noise; Words like 'the', 'a', 'an', 'of', 'for', etc remain in the analysis.

## Pronouns

Related to proper counting and stopwords are proper nouns. In a script or novel, the names of the characters of the story will show up a disporportionate amount of the time. With a large enough corpus this becomes moot because names common to the era will naturally show up more than modern names. However, these corpora aren't large enough for this averaging of character names. The same is true for place names. The location the script is set has a higher likelihood of being mentioned.

## Ratio impact

As can be seen in the final results dataframe, the high ratios have a much larger impact on my ranking than the lower numbers. This means that including words that were rare in the 1960s has a much bigger impact on the ranking than excluding words that were common.

### Repetition

The authentic 1960s corpus includes many, many The Twilight Zone episodes. Most, if not all, of The Twilight Zone episodes start with the same introduction. This means that words like 'traveling', 'another', 'dimension', 'sight', 'sound', 'mind', and 'journey' are disproportionately represented. An improvement to the analysis would be to account for and remove this repetition so that it's only represented once in the frequencies.
