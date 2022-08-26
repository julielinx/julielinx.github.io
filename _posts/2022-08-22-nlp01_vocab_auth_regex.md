---
title: "Entry NLP1: Corpus Cleaning with RegEx"
categories:
  - Blog
tags:
  - nlp
  - regex
---

Recently I've been working my way through [one of the older versions](https://courses.cs.washington.edu/courses/cse140/13wi/) of [CSE 140](http://courses.cs.washington.edu/courses/cse160/) (now CSE 160) offered at the University of Washington. [Homework 8](https://courses.cs.washington.edu/courses/cse140/13wi/homework/hw8/assignment.html) is a nice exercise that requires natural language processing (NLP) and analysis.

The homework as specified in the directions exlcudes specific NLP techniques that I generally include, specifically case insensitivity and punctuation removal. The example results for the solution also imply the use of tuples, whereas I prefer dataframes. Due to these alterations (my solution would receive a failing grade if submitted), I feel comfortable posting it to showcase some natural language processing techniques.

The series of posts walking through this linguistic analysis example is significantly different than my other posts. I'll walk through my solution with samples of the output as I create the code.

See the linked notebook for the condensed final solution minus any commentary. *Note,* code results have been truncated for brevity. For the full results, see the [accompanying notebook](https://github.com/julielinx/datascience_diaries/blob/master/nlp/01_vocab_auth_regex.ipynb).


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

## Read in file

Obviously, the first thing to do is read in the data. Most of the files are `.srt` format, which has a package `pysrt` for easy reading. The homework directions forbid using this package, and I found I appreciated the control I had in reading in files without it.

The standard and very basic way to read in a file is to use `open()`. When using this function I'll read in the file line by line. This allows me to remove lines that aren't pertinent to the analysis before having to store them in memory.

```python
test_file = os.path.join(os.getcwd(), 'data', '1960s', 'The Twilight Zone - 3x17 - One More Pallbearer.srt')
with open(test_file, 'r') as l:
    for line in l:
        print(line, end='')
```

    0
    00:00:01,000 --> 00:00:04,000
    Downloaded From www.AllSubs.org
    
    1
    00:00:00,105 --> 00:00:05,000
    Shared by http://DJJ.HOME.SAPO.PT/
    
    1
    00:00:11,378 --> 00:00:13,880
    You're traveling
    through another dimension-
    
    2
    00:00:13,915 --> 00:00:17,017
    a dimension not only of sight
    and sound, but of mind,

## Remove lines/characters

The homework directions list the following things to remove:

- Any line that contains only numeric characters ('0', '1', ... '9')
- Any line that contains the string '-->'
- Blank lines
- The following chracters/character groups:
    - '\<i>'
    - '\</i>'
    - '\<font color=#00FF00>'
    - '\<font color=#00FFFF>'
    - '\<font color="#00ff00">'
    - '\<font color="#ff0000">'
    - '\</font>'
    - '#'
    - '-'
    - '('
    - ')'
    - '\xe2\x99\xaa'
    - 'www.AllSubs.org'
    - 'http://DJJ.HOME.SAPO.PT/'
    - 'Downloaded'
    - 'Shared'
    - 'Sync'
    - 'www.addic7ed.com'
    - 'n17t01'
    - '"'
    - '\n'
    
I'll be using the regex python library `re` to handle finding these things. Most of the regex patterns and logic used in this notebook can be found on the cheatsheet of [regexr.com](https://regexr.com/).
    
Looking at the initial sample I loaded in, the reason for the first three items is easily apparent;

- Each line block is numbered
- The second line of the block is two timestamps separated by '-->'
    - A quick internet search reveals that the `srt` file format is one of the most common file formats for subtitling and/or captioning. The timestamps are the start and stop times of the lines that follow within the line block
- There is an empty line between each line block

```python
with open(test_file, 'r') as l:
    for line in l:
        print(line, end='')
```

    0
    00:00:01,000 --> 00:00:04,000
    Downloaded From www.AllSubs.org
    
    1
    00:00:00,105 --> 00:00:05,000
    Shared by http://DJJ.HOME.SAPO.PT/
    
    1
    00:00:11,378 --> 00:00:13,880
    You're traveling
    through another dimension-
    
    2
    00:00:13,915 --> 00:00:17,017
    a dimension not only of sight
    and sound, but of mind,

Fortunately, the first two things can be identified using a single line of regex. The regex pattern '\d+' looks for patterns that only include digits (i.e. no letters).

```python
with open(test_file, 'r') as l:
    for line in l:
        if re.match('\d+', line):
            print(line, end='')
```

    0
    00:00:01,000 --> 00:00:04,000
    1
    00:00:00,105 --> 00:00:05,000
    1
    00:00:11,378 --> 00:00:13,880
    2
    00:00:13,915 --> 00:00:17,017

Now that I've verified that the regex pattern only identifies the lines I'm interested in, I can use it to remove those lines by returning the ones that don't match the criteria.


```python
with open(test_file, 'r') as l:
    for line in l:
        if not re.match('\d+', line):
            print(line, end='')
```

    Downloaded From www.AllSubs.org
    
    Shared by http://DJJ.HOME.SAPO.PT/
    
    You're traveling
    through another dimension-
    
    a dimension not only of sight
    and sound, but of mind,

Alternately, instead of excluding the lines using `not` as part of the `if` statement, I can also exclude those lines using the regex pattern by changing it to a set that includes the `^` not character. 

```python
with open(test_file, 'r') as l:
    for line in l:
        if re.match('[^\d+]', line):
            print(line, end='')
```

    Downloaded From www.AllSubs.org
    
    Shared by http://DJJ.HOME.SAPO.PT/
    
    You're traveling
    through another dimension-
    
    a dimension not only of sight
    and sound, but of mind,

With those lines removed, it's easy to see why "Downloaded", "www.AllSubs.org", "Shared", and "http://DJJ.HOME.SAPO.PT/" are on the list of character groups to remove. However, blank lines are thrid on the list, so I'm going to get rid of those first.

I admit, I had to search the internet for the [regex pattern to remove blank lines](https://www.codegrepper.com/code-examples/javascript/regex+empty+line). The pattern is the unintuitive `^(?!\s*$).+`.

- `^` indicates the start of the string (when included as part of a set it means `not`, see the digit removal above for an example)
- `()` denotes a group
- `?!` is a negative lookahead 
- `\s` identified white space
- `*` matches 0 or more of the charcter it follows
- `$` indicates the end of the string
- `.` is a wildcard, it matches any character except for newline
- `+` matches 1 or more of the character it follows

Testing it on the sample file, I can confirm it does exactly what's needed.

```python
with open(test_file, 'r') as l:
    for line in l:
        if re.match('^(?!\s*$).+', line):
            print(line, end='')
```

    0
    00:00:01,000 --> 00:00:04,000
    Downloaded From www.AllSubs.org
    1
    00:00:00,105 --> 00:00:05,000
    Shared by http://DJJ.HOME.SAPO.PT/
    1
    00:00:11,378 --> 00:00:13,880
    You're traveling
    through another dimension-
    2
    00:00:13,915 --> 00:00:17,017
    a dimension not only of sight
    and sound, but of mind,

Now that I've got the first three big items knocked off the list, I can start addressing the special characters/character groups to remove.

I'm going to start with the ones jumping out at me: "Downloaded", "www.AllSubs.org", "Shared", and "http://DJJ.HOME.SAPO.PT/".

A quick internet search shows that the internet wasn't really a thing until 1969, and then only in its infancy. The world wide web (www) wasn't invented until 1989 and not released to the public until 1993. All of which means that any lines including "www." or "http:" are metadata that I need to remove (if that isn't true, then the script has bigger problems that incorrect vocabulary).

```python
with open(test_file, 'r') as l:
    for line in l:
        if re.match('(.*www.*)|(.*http:*)', line):
            print(line, end='')
```

    Downloaded From www.AllSubs.org
    Shared by http://DJJ.HOME.SAPO.PT/
    Downloaded From www.AllSubs.org

This single regex line knocks quite a few items off the list:

- 'www.AllSubs.org'
- 'http://DJJ.HOME.SAPO.PT/'
- 'Downloaded'
- 'Shared'
- 'www.addic7ed.com'

Knocking out 5 items at once feels good, so next I'll tackle another big chunk that can be easily identified using regex:

- '\<i>'
- '\</i>'
- '\<font color=#00FF00>'
- '\<font color=#00FFFF>'
- '\<font color="#00ff00">'
- '\<font color="#ff0000">'
- '\</font>'

However, to see any of them, I have to switch my sample file. I switched to one of the Pan Am scripts. Looking at the raw file, the `<i>` pattern is readily available in the first line block.

```python
test_file2 = test_file_path = os.path.join(os.getcwd(), 'data', '21st-century', 'Pan_Am', 'Pan.Am.S01E08.srt')
with open(test_file2, 'r') as l:
    for line in l:
        print(line, end='')
```

    ﻿1
    00:00:01,461 --> 00:00:02,729
    <i>Previously on "Pan Am"...</i>
    
    2
    00:00:02,796 --> 00:00:06,064
    Let's keep it in New York,
    Ginny. Monte Carlo was a lark.
    
    3
    00:00:06,348 --> 00:00:08,949
    It's likely to be
    a long trip.

```python
with open(test_file2, 'r') as l:
    for line in l:
        if re.match('</?i>|</?font.*>', line):
            print(line, end='')
```

    <i>Previously on "Pan Am"...</i>
    <i>I</i> am in charge in the air,
    <i>Attention passengers
    <i>L'aéroport est fermé.</i>
    <i>Nous ne pouvons pas...</i>
    <i>Port-au-Prince,</i>
    <i>nous avons un passager mourant.</i>
    <i>Vous n'avez pas l'autorisation.</i>    

Unfortunately, there are no examples of `<font>`. But if the generic pattern works for `<i>` it should also work for `<font>`.

Unlike the lines before which were entirely metadata, the words in these lines should be included in the analysis. As such, I don't want to remove the lines entirely, just a specific group of characters. To do this, I switch from `re.match` to `re.sub`. 

```python
with open(test_file2, 'r') as l:
    for line in l:
        line = re.sub('</?i>|</?font.*>', '', line)
        print(line, end='')
```

    ﻿1
    00:00:01,461 --> 00:00:02,729
    Previously on "Pan Am"...
    
    2
    00:00:02,796 --> 00:00:06,064
    Let's keep it in New York,
    Ginny. Monte Carlo was a lark.
    
    3
    00:00:06,348 --> 00:00:08,949
    It's likely to be
    a long trip.

Voilà, another 7 items knocked off the list.

Of the original items from the characters/character group list, I'm left with:

- '#'
- '-'
- '('
- ')'
- '\xe2\x99\xaa'
- 'Sync'
- 'n17t01'
- '"'
- '\n'

Of these, most are special characters (#, -, (, ), \n, "). I'm going to handle these in another part of the analysis where I can remove all special characters at the same time (also including characters like !, ., ?, etc).

The '\xe2\x99\xaa' item is a character encoding problem. This homework lesson must have been created before improved character encoding handling because this series of characters didn't show up in my earilier unigram analysis. I'm going to assume the 'latin-1' encoding required to process all the files (default 'utf-8' handles most, but not all) appropriately handles this.

That leaves me with  'Sync' and 'n17t01'. I wasn't able to find the file(s) that include 'n17t01', but the Pan Am sample has 'Sync'.


```python
with open(test_file2, 'r') as l:
    for line in l:
        if re.match('Sync', line):
            print(line, end='')
```

    Sync and corrected by dr.jackson
    Sync and corrected by dr.jackson

Since this appears to be metadata, I'm going to remove the whole line so that the other words aren't included in the analysis either. A quick internet search indicates that 'n17t01' also tends to be included in a line 'Sync and corrections by n17t01'.

To ensure I don't remove all lines with 'sync', inadventently removing lines that should be analyzed, I'll search for a partial phrase. The pattern 'Sync and correct*' should match both "Sync and corrected by" and "Sync and corrections by", which should remove all the 'sync' lines I don't want while leaving any lines that include 'sync' as part of the text.

```python
with open(test_file2, 'r') as l:
    for line in l:
        if re.match('Sync and correct*', line):
            print(line, end='')
```

    Sync and corrected by dr.jackson
    Sync and corrected by dr.jackson

This concludes the pattern matching section. Now all I have to do is put it all together:

```python
with open(test_file, 'r') as l:
    for line in l:
        if (re.match('[^\d+]', line)
           ) and (re.match('^(?!\s*$).+', line)
                  ) and not (re.match('(.*www.*)|(.*http:*)', line)
                            ) and not (re.match('Sync and correct*', line)):
            line = re.sub('</?i>|</?font.*>', '', line)
            print(line, end='')
```

    You're traveling
    through another dimension-
    a dimension not only of sight
    and sound, but of mind,
    a journey into a wondrous land
    whose boundaries
    are that of imagination.
    Your next stop,
    the twilight zone.

```python
with open(test_file2, 'r') as l:
    for line in l:
        if (re.match('[^\d+]', line)
           ) and (re.match('^(?!\s*$).+', line)
                  ) and not (re.match('(.*www.*)|(.*http:*)', line)
                            ) and not (re.match('Sync and correct*', line)):
            line = re.sub('</?i>|</?font.*>', '', line)
            print(line, end='')
```

    ﻿1
    Previously on "Pan Am"...
    Let's keep it in New York,
    Ginny. Monte Carlo was a lark.
    It's likely to be
    a long trip.

Interestingly, the first numeric only line in the Pam Am file wasn't removed. Since the regex pattern handles all the others correctly, I'm going to leave this artifact for the time being.

## Store data

Now that I can read in only the lines I want, I need to store the results in some kind of data structure. The structures I'd consider for this exercise are lists, concatenated strings, a dictionary, or a dataframe.

Having read through the full homework instructions, the bigram processing is supposed to treat all lines within a script as one corpus, but not lines between scripts. So Twilight episode 1 has to be processed separately from Twilight episode 2. This means each script must be processed into n-grams before combining all the results to grab n-gram frequencies.

For the first part (treat all lines within a script as one corpus), a concatenated string is the best option. This allows me to do all processing (like removing special characters, transforming all words to lowercase,  removing stopwords, and transforming to the appropriate n-gram) at the same time.

The second part (don't create n-grams between scripts), means I have to store each script's corpus separately, then combine them once they've been transformed into n-grams.

My prefered method of doing this is to use a dictionary. Although not my favorite when I need to look up or sort by both the key and the value (i.e. I won't be using dictionaries for the frequency/analysis part of this homework), the dictionary is a very versitle data structure that can hold other data structures. For this exercise, I'll store the individual script corpora as values in the dictionary.

One of the major benefits of this method is the ability to name another data structure for reference (in this case the corpus from the Dr Strangelove script would be named "Dr.Strangelove". However, the majority of the script names are a problem (I refuse to type something as long as "The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room" to access a particular item), but if I automate the handling, I shouldn't have to manually enter the script name. However, if I need to know the name of a particular script, I can still look it up.

The first step of this is to store the lines as a single text corpus.

```python
test_file = os.path.join(os.getcwd(), 'data', '1960s', 'The Twilight Zone - 3x17 - One More Pallbearer.srt')
corpus = ''
with open(test_file, 'r', encoding='latin-1') as l:
    for line in l:
        if (re.match('[^\d+]', line)
           ) and (re.match('^(?!\s*$).+', line)
                  ) and not (re.match('(.*www.*)|(.*http:*)', line)
                            ) and not (re.match('Sync and correct*', line)):
            line = re.sub('</?i>|</?font.*>', '', line)
            corpus = corpus + ' ' + line
corpus[:1000]
```

    " You're traveling\n through another dimension-\n a dimension not only of sight\n and sound, but of mind,\n a journey into a wondrous land\n whose boundaries\n are that of imagination.\n Your next stop,\n the twilight zone.\n She's all set,\n mr. Radin.\n How about the\n sound system?\n You check that out?\n She's all\n ready to go.\n I don't know where you\n got your sound effects\n but you'd swear\n a bomb was exploding.\n I mean a big bomb.\n That's precisely the way\n it's supposed to sound.\n That about do it,\n mr. Radin?\n That about does it.\n You got quite\n a setup here.\n This part of\n the illusion too?\n No, this room is\n not an illusion.\n I venture to guess\n that it's the best\n designed bomb shelter\n on the face of the\n earth- who knows?\n The hydrogen bomb\n is not an illusion.\n But tonight it's\n for gags, huh?\n Something of the sort.\n A practical joke,\n let's say.\n You can say\n that again.\n When they start\n those sound effects\n and that stuff\n on the screen\n you'd swear the world\n was getting blasted.\n"

This is the same code from before, but instead of printing the line, we store it in a string. Make note of the extra space I add when adding a new line to the string. If this is left out, the lines run into each other, which will negatively impact the analysis.

For example, as can be seen below, instead of getting 'traveling' and 'through' as separate words, then would be combined once the '\n' character is removed: 'travelingthrough'.

```python
corpus = ''
with open(test_file, 'r', encoding='latin-1') as l:
    for line in l:
        if (re.match('[^\d+]', line)
           ) and (re.match('^(?!\s*$).+', line)
                  ) and not (re.match('(.*www.*)|(.*http:*)', line)
                            ) and not (re.match('Sync and correct*', line)):
            line = re.sub('</?i>|</?font.*>', '', line)
            corpus = corpus + line
corpus[:1000]
```

    "You're traveling\nthrough another dimension-\na dimension not only of sight\nand sound, but of mind,\na journey into a wondrous land\nwhose boundaries\nare that of imagination.\nYour next stop,\nthe twilight zone.\nShe's all set,\nmr. Radin.\nHow about the\nsound system?\nYou check that out?\nShe's all\nready to go.\nI don't know where you\ngot your sound effects\nbut you'd swear\na bomb was exploding.\nI mean a big bomb.\nThat's precisely the way\nit's supposed to sound.\nThat about do it,\nmr. Radin?\nThat about does it.\nYou got quite\na setup here.\nThis part of\nthe illusion too?\nNo, this room is\nnot an illusion.\nI venture to guess\nthat it's the best\ndesigned bomb shelter\non the face of the\nearth- who knows?\nThe hydrogen bomb\nis not an illusion.\nBut tonight it's\nfor gags, huh?\nSomething of the sort.\nA practical joke,\nlet's say.\nYou can say\nthat again.\nWhen they start\nthose sound effects\nand that stuff\non the screen\nyou'd swear the world\nwas getting blasted.\nThat's the idea.\nI have three guests\ncoming this eve"

This processing needs to be done to every single file provided by the instructor. The best way to do this is to turn the code into a function. Now that I have all of the pieces to process the text, I'm ready to create my first function.

*Side note:* This function is different than the one I originally created in an earlier homework8 notebooks (I hadn't read the full homework assignment and understood the downstream reprecussions discussed above re: unigram vs bigram handling. Also, I was trying to complete the assignment without using the special character and stopword handling from the nltk library).

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
```

This concludes the first post in the series.