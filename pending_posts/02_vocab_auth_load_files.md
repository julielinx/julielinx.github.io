---
title: "Entry NLP1: Corpus Cleaning with RegEx"
categories:
  - Blog
tags:
  - nlp
  - reading files
---
# Entry NLP2: Load All Files in a Directory

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

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/julie.fisher/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True



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
    The Twilight Zone - s05e01 - In Praise of Pip.srt
    The Twilight Zone - 4x13 - The New Exhibit.srt
    The Twilight Zone - 3x21 - Kick the Can.srt
    The Twilight Zone - 3x30 - Hocus-Pocus and Frisby.srt
    The Twilight Zone - 3x16 - Nothing in the Dark.srt
    The Twilight Zone - s05e12 - Ninety Years Without Slumbering.srt
    The Twilight Zone - 2x08 - The Lateness of the Hour.srt
    The Twilight Zone - 4x11 - The Parallel.srt
    The Twilight Zone - s05e35 - The Fear.srt
    The Twilight Zone - 2x09 - The Trouble with Templeton.srt
    The Twilight Zone - s05e09 - Probe 7, Over and Out.srt
    The Twilight Zone - s05e03 - Nightmare at 20,000 Feet.srt
    The Twilight Zone - 4x16 - On Thursday We Leave for Home.srt
    The Twilight Zone - 3x36 - Cavender Is Coming.srt
    The Twilight Zone - 4x06 - Death Ship.srt
    The Twilight Zone - 2x21 - The Prime Mover.srt
    The Twilight Zone - s05e29 - The Jeopardy Room.srt
    The Twilight Zone - 3x35 - I Sing the Body Electric.srt
    The Twilight Zone - 2x01 - King Nine Will Not Return.srt
    The Twilight Zone - 2x20 - Static.srt
    The Twilight Zone - 2x06 - Eye of the Beholder.srt
    The Twilight Zone - s05e04 - A Kind of a Stopwatch.srt
    The Twilight Zone - 4x01 - In His Image.srt
    The Twilight Zone - 3x24 - To Serve Man.srt
    The Twilight Zone - s05e14 - You Drive.srt
    The Twilight Zone - s05e32 - Mr. Garrity and the Graves.srt
    The Twilight Zone - s05e27 - Sounds and Silences.srt
    The Twilight Zone - 3x10 - The Midnight Sun.srt
    The Twilight Zone - s05e19 - Night Call.srt
    The Twilight Zone - 2x19 - Mr. Dingle, the Strong.srt
    The Twilight Zone - 4x02 - The Thirty-Fathom Grave.srt
    The Twilight Zone - s05e10 - The 7th Is Made up of Phantoms.srt
    The Twilight Zone - 2x10 - A Most Unusual Camera.srt
    The Twilight Zone - 3x34 - Young Man's Fancy.srt
    The Twilight Zone - 2x23 - A Hundred Yards over the Rim.srt
    The Twilight Zone - 3x11 - Still Valley.srt
    Dr.Strangelove.srt
    The Twilight Zone - 2x28 - Will the Real Martian Please Stand Up.srt
    The Twilight Zone - s05e06 - Living Doll.srt
    The Twilight Zone - 4x09 - Printer's Devil.srt
    The Twilight Zone - s05e15 - The Long Morrow.srt
    The Twilight Zone - 2x11 - The Night of the Meek.srt
    The Twilight Zone - s05e08 - Uncle Simon.srt
    The Twilight Zone - 2x13 - Back There.srt
    The Twilight Zone - 4x18 - The Bard.srt
    The Twilight Zone - s05e28 - Caesar and Me.srt
    The Twilight Zone - s05e16 - The Self-Improvement of Salvadore Ross.srt
    The Twilight Zone - 2x22 - Long Distance Call.srt
    The Twilight Zone - s05e20 - From Agnes - with Love.srt
    The Twilight Zone - 3x28 - The Little People.srt
    The Twilight Zone - 3x19 - The Hunt.srt
    The Twilight Zone - s05e17 - Number 12 Looks Just Like You .srt
    The Twilight Zone - 2x02 - The Man in the Bottle.srt
    The Twilight Zone - s05e18 - Black Leather Jackets.srt
    The Twilight Zone - 3x27 - Person or Persons Unknown.srt
    The Twilight Zone - s05e11 - A Short Drink from a Certain Fountain.srt
    The Twilight Zone - 4x12 - I Dream of Genie.srt
    The Twilight Zone - 3x09 - Deaths-Head Revisited.srt
    The Twilight Zone - 3x37 - The Changing of the Guard.srt
    The.Hustler.srt
    The Twilight Zone - 4x03 - Valley of the Shadow.srt
    The Twilight Zone - s05e24 - What's in the Box.srt
    The Twilight Zone - s05e31 - The Encounter.srt
    The Twilight Zone - s05e30 - Stopover in a Quiet Town.srt
    The Twilight Zone - s05e23 - Queen of the Nile.srt
    The Twilight Zone - 3x15 - A Quality of Mercy.srt
    The Twilight Zone - 3x07 - The Grave.srt
    The.Apartment.srt
    The Twilight Zone - 2x24 - The Rip Van Winkle Caper.srt
    The Twilight Zone - 3x20 - Showdown with Rance McGrew.srt
    The Twilight Zone - s05e25 - The Masks.srt
    The Twilight Zone - 4x08 - Miniature.srt
    The Twilight Zone - 3x02 - The Arrival.srt
    The Twilight Zone - 4x10 - No Time Like the Past.srt
    The Twilight Zone - 2x16 - A Penny for Your Thoughts.srt
    The Twilight Zone - 3x08 - It's a Good Life.srt
    Lover Come Back.srt
    The Twilight Zone - 2x12 - Dust.srt
    The Twilight Zone - 3x32 - The Gift.srt
    The Twilight Zone - 2x26 - Shadow Play.srt
    The Twilight Zone - 3x33 - The Dummy.srt
    The Twilight Zone - s05e26 - I am the Night - Color Me Black.srt
    The Twilight Zone - 3x31 - The Trade-Ins.srt
    The Twilight Zone - s05e02 - Steel.srt
    The Twilight Zone - 3x18 - Dead Man's Shoes.srt
    The Twilight Zone - 2x17 - Twenty Two.srt
    The Twilight Zone - 2x18 - The Odyssey of Flight 33.srt
    The Twilight Zone - 2x04 - A Thing About Machines.srt
    The Twilight Zone - 3x26 - Little Girl Lost.srt
    The Twilight Zone - 4x17 - Passage on the Lady Anne.srt
    The Twilight Zone - 3x23 - The Last Rites of Jeff Myrtlebank.srt
    Lilies.of.the.Field.srt
    The Twilight Zone - s05e34 - Come Wander with Me.srt
    The Twilight Zone - 3x12 - The Jungle.srt
    The Twilight Zone - 2x14 - The Whole Truth.srt
    The Twilight Zone - s05e05 - The Last Night of a Jockey.srt
    The Twilight Zone - 2x07 - Nick of Time.srt
    The Twilight Zone - 4x04 - He's Alive.srt
    The Twilight Zone - 3x06 - The Mirror.srt
    The Twilight Zone - 3x22 - A Piano in the House.srt
    The Twilight Zone - 3x29 - Four O'Clock.srt
    The Twilight Zone - 4x14 - Of Late I Think of Cliffordville.srt
    The Twilight Zone - 4x07 - Jess-Belle.srt
    The Twilight Zone - s05e33 - The Brain Center at Whipple's.srt
    The Twilight Zone - 2x05 - The Howling Man.srt
    The Twilight Zone - 3x25 - The Fugitive.srt
    The Twilight Zone - 2x25 - The Silence.srt
    The Twilight Zone - 2x27 - The Mind and the Matter.srt
    The Twilight Zone - 4x15 - The Incredible World of Horace Ford.srt
    The Twilight Zone - s05e07 - The Old Man in the Cave.srt
    The Twilight Zone - 3x14 - Five Characters in Search of an Exit.srt
    The Twilight Zone - s05e36 - The Bewitchin' Pool.srt
    The Twilight Zone - 3x03 - The Shelter.srt
    The Twilight Zone - s05e21 - Spur of the Moment.srt
    The Twilight Zone - 2x29 - The Obsolete Man.srt
    The Twilight Zone - s05e13 - Ring-A-Ding Girl.srt
    

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
