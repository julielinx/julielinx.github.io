---
title: "Entry NLP2: Load All Files in a Directory"
categories:
  - Blog
tags:
  - nlp
  - reading files
---

In the previous entry, I figured out how to process individual files, removing many of the items on the "Remove lines/characters" list specified in the homework. However, there's no way I want to individually list out each of the files. I need a way to get my code to consider all files within a directory.

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

The `read_script` function was the output of the last entry.

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

## Load all files in a directory

My original understanding of the assignment was that I was supposed to compare all scripts in the '1960s' directory with all the scripts in the '21st-century' directory. However, having re-read the later portion of the assignment, my understanding is that the purpose is to compare all scripts in the '1960s' directory against the different shows/movie in the '21st-century' directory.

Listing out all files within a directory is pretty straight forward with the `os` library.

- `scandir` scans the directory
- `path.join` joins parts of the file path so that handling is operating system agnostic
    - Mac uses '/'
    - PC uses '\'
- `getcwd` gets the current working directory
    - This allows the file structure to be independent of the upper file structure
    - This independence allows the .py script to be transferable between computers and/or users

```python
# file_path = 
for thing in os.scandir(os.path.join(os.getcwd(), 'data', '1960s')):
    print(thing.name)
```

    The Twilight Zone - 3x17 - One More Pallbearer.srt
    The Twilight Zone - 3x05 - A Game of Pool.srt
    The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room.srt
    The Twilight Zone - 4x05 - Mute.srt
    The Twilight Zone - 3x04 - The Passersby.srt

To load the data for the 1960s scripts I can do something like this:

```python
file_path = os.path.join(os.getcwd(), 'data', '1960s')
test_dict = {}
for thing in os.scandir(file_path):
    test_dict[thing.name] = read_script(f'{file_path}/{thing.name}')
```

Now I verify that the data loaded as expected: name of the script as the key, concatenated string as the value (yes, I absolutely copy and pasted the name of the first script to get its corpus).

```python
list(test_dict.keys())[:5]
```

    ['The Twilight Zone - 3x17 - One More Pallbearer.srt',
     'The Twilight Zone - 3x05 - A Game of Pool.srt',
     'The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room.srt',
     'The Twilight Zone - 4x05 - Mute.srt',
     'The Twilight Zone - 3x04 - The Passersby.srt']

```python
test_dict['The Twilight Zone - 3x17 - One More Pallbearer.srt'][:500]
```

    " You're traveling\n through another dimension-\n a dimension not only of sight\n and sound, but of mind,\n a journey into a wondrous land\n whose boundaries\n are that of imagination.\n Your next stop,\n the twilight zone.\n She's all set,\n mr. Radin.\n How about the\n sound system?\n You check that out?\n She's all\n ready to go.\n I don't know where you\n got your sound effects\n but you'd swear\n a bomb was exploding.\n I mean a big bomb.\n That's precisely the way\n it's supposed to sound.\n That about do it,\n mr"

The challenge is the '21st-century' directory: it contains four subdirectories. The homework instructions say to accept a directory and just load the files within it (so I'd have to run the script four times to analyze all the 21st century corpora), but since I'm going off script on an extended detour to improve the analysis, I want to be able to run this thing once and get the analyses for all four test corpora.

What I need is a nested dictionary. The goal is something like this:
>{</br>
>Mad_Men:{script_1: corpus_1, script_2: corpus_2, ..., script_n: corpus_n},</br>
>Pan_Am: {script_1: corpus_1, script_2: corpus_2, ..., script_n: corpus_n},</br>
The_Kennedys: {script_1: corpus_1, script_2: corpus_2, ..., script_n: corpus_n},</br>
X-Men_First_Class: {script1: corpus1}</br>
>}

In my previous solution, I used recursion to work through any subdirectories until I grabbed the file paths for all files the specified directory and its subdirectories then returned that as a list.

```python
def list_filepaths(append_list, file_path = os.getcwd()):
    for thing in os.scandir(file_path):
        if thing.is_dir():
            new_path = os.path.join(file_path, thing.name)
            list_filepaths(append_list, new_path)
        elif thing.is_file:
            append_list.append(f'{file_path}/{thing.name}')
    return append_list
```

I improved upon that concept when I added the `read_script` function to the example code:

```
file_path = os.path.join(os.getcwd(), 'data', '1960s')
test_dict = {}
for thing in os.scandir(file_path):
    test_dict[thing.name] = read_script(f'{file_path}/{thing.name}')
```

A few minor edits to the original recursive function gives me what I want. [This StackOverflow answer](https://stackoverflow.com/a/48382262) gave me the key I needed to recursively create the dictionary structure: add the dictionary as a parameter, then create a new level to feed to the recursion:

```python
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

```python
print(list(unilayer_dict.keys())[:5]) # First 5 scripts in the 1960s directory
print('\n')
print(list(unilayer_dict.keys())[0], 'sample text:') # Name of the first script in the list
print(unilayer_dict[list(unilayer_dict.keys())[0]][:500]) # 500 character sample of the first script
```

    ['The Twilight Zone - 3x17 - One More Pallbearer.srt', 'The Twilight Zone - 3x05 - A Game of Pool.srt', 'The Twilight Zone - 2x03 - Nervous Man in a Four Dollar Room.srt', 'The Twilight Zone - 4x05 - Mute.srt', 'The Twilight Zone - 3x04 - The Passersby.srt']
    
    
    The Twilight Zone - 3x17 - One More Pallbearer.srt sample text:
     You're traveling
     through another dimension-
     a dimension not only of sight
     and sound, but of mind,
     a journey into a wondrous land
     whose boundaries
     are that of imagination.
     Your next stop,
     the twilight zone.
     She's all set,
     mr. Radin.
     How about the
     sound system?
     You check that out?
     She's all
     ready to go.
     I don't know where you
     got your sound effects
     but you'd swear
     a bomb was exploding.
     I mean a big bomb.
     That's precisely the way
     it's supposed to sound.
     That about do it,
     mr

```python
file_path = os.path.join(os.getcwd(), 'data', '21st-century')
bilayer_dict = load_files_to_dict(file_path, {})
```

```python
print(list(bilayer_dict.keys())) # List of the subdirectories in the 21st-century directory
print(list(bilayer_dict.keys())[0]) # Name of the first subdirectory in the list
print(list(bilayer_dict[list(bilayer_dict.keys())[0]].keys())) # List of the scripts of the first subdirectory within the subdirectory list
print('\n')
print(list(bilayer_dict[list(bilayer_dict.keys())[0]].keys())[0], 'sample text')
print(bilayer_dict[list(bilayer_dict.keys())[0]][list(bilayer_dict[list(bilayer_dict.keys())[0]].keys())[0]][:500]) # 500 character sample of the script
```

    ['Pan_Am', 'Mad_Men', 'X-Men_First_Class', 'The_Kennedys']
    Pan_Am
    ['Pan.Am.S01E09.srt', 'Pan.Am.S01E08.srt', 'Pan.Am.S01E05.srt', 'Pan.Am.S01E11.srt', 'Pan.Am.S01E10.srt', 'Pan.Am.S01E04.srt', 'Pan.Am.S01E12.srt', 'Pan.Am.S01E06.srt', 'Pan.Am.S01E07.srt', 'Pan.Am.S01E13.srt', 'Pan.Am.S01E03.srt', 'Pan.Am.S01E02.srt', 'Pan.Am.S01E14.srt', 'Pan.Am.S01E01.srt']
    
    
    Pan.Am.S01E09.srt sample text
     Previously on "Pan Am"...
     Look, I get to see the world,
     Sam.
     When was the last time
     you left the village?
     I don't need to see the world
     to change it.
     - Marry me!
     - I can't say yes now.
     Pan Am stewardess can travell all
     around the world without suspicion.
     You volunteered for this.
     They will let you out.
     Are you going my way?
     Sometimes the stars align.
     You're different
     from other girls.
     Thank you.
     And democracy is not perfect.
     You're casting a shadow,
     Kate.
     I take it you miss
    
Ta-da. I can now easily load and store all of the 1960s and 21st century corpora while only having to specify two file paths.