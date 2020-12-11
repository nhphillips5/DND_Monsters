# DND_Monsters
Predicting a Monsters Challenge Rating

As a full fledged nerd, not only am I fluent in code, but also in Dungeons and 
Dragons (or DnD as all the "cool" kids say). I've played the game since I was
a young child of 5. I always loved looking through the pictures of the fiend 
folio and the monster manuals. As such when I usually think of ways machine 
learning can improve my life I often think of games such as DnD.

So I wondered, could a ML model predict the challenge rating of a monster in 
DnD? For those of you that are not fluent in DnD, a challenge rating is given
to each monster in DnD to help the Dungeon Master (the player of the game that 
is the main story teller and throws obstacles, monsters and other threats at the
other players). The image below shows the stat block for a dragon with the
challenge rating highlighted.

![Challenge Rating Image](https://4.bp.blogspot.com/-RS6NaXZPb6A/V5vQO-IFt9I/AAAAAAAAIZI/3JqykSpr1k4oBkfYMpmoQs0Vq3mLaNSxwCLcB/s1600/challengerating.jpg)

This would be a great help to me because I'm constantly making my own monsters 
for my own players. The issue is I have no idea how hard this will be for them.
This model would be able to let me know whether I'm about to kill my players or
give them an easy kill.

The data that I got for this project was from the lovely website [Open5e](https://open5e.com/).
I went through their Live API and downloaded each of the json files. Let's jump
into the code! I first import the modules I'll need, then read in the data:

```python
import numpy as np
import pandas as pd
import re
import math
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
        classification_report)

#Files I will be reading in
filenames = ["monsters/mon1.json",
             "monsters/mon2.json",
             "monsters/mon3.json",
             "monsters/mon4.json",
             "monsters/mon5.json",
```

There's a total of 22 of those files so I won't put the whole thing.

```python
monsters = []
for f in filenames:
    monsters.append(pd.read_json(f))

mon = pd.DataFrame()

mon = mon.append(monsters)

mon = mon['results'].apply(pd.Series)
```

Now that I've read in all the data I need to clean it up a bit. First I'll drop
all of the columns that I'm not going to use at all. Some of these are useless
and some just had data that would take to long to clean to make them useful. 
In the future I'll probably come back and carve them up with regular expressions.

```python
mon = mon.drop(['slug', 'subtype','group', 'armor_desc', 'hit_dice', 'speed',
    'skills', 'damage_vulnerabilities', 'damage_resistances',
    'damage_immunities', 'condition_immunities', 'senses', 'languages',
    'legendary_desc', 'img_main', 'document__slug', 'document__title',
    'document__license_url'], axis = 1)
```

I kept a few of the features that contain nested lists because I can pull out
some information from them. Using pythons `len(x)` function I can get how many 
items are nested and it might prove useful. After applying that, I remove the
original columns.

```python
mon['num_actions'] = mon['actions'].apply(lambda x: len(x))
mon['num_reactions'] = mon['reactions'].apply(lambda x: len(x))
mon['num_legendary_actions'] = mon['legendary_actions'].apply(lambda x: len(x))
mon['num_special_abilities'] = mon['special_abilities'].apply(lambda x: len(x))
mon['num_spells'] = mon['spell_list'].apply(lambda x: len(x))

#Remove the Original Columns
mon = mon.drop(['actions', 'reactions', 'legendary_actions',
    'special_abilities', 'spell_list'], axis = 1)
```

Then to further clean the data I needed to get the challenge rating into a
numeric type. The issue is that it has values like '1/8', '1/4', '1/2'. I wanted
to keep their ordinal behavior so I wrote a function to convert them to integers
while raising the higher values accordingly.

```python
#Converting Function
def cr_converter(x):
    if x == '1/8':
        x = 1
        return x
    elif x == '1/4':
        x = 2
        return x
    elif x == '1/2':
        x = 3
        return x
    elif x == '0':
        x = 0
        return x
    else:
        x = int(x) + 3
        return x

#Make sure it works
print(cr_converter('0'))
print(cr_converter('1/8'))
print(cr_converter('1/4'))
print(cr_converter('1/2'))
print(cr_converter('1'))
print(cr_converter('30'))

#Apply it
mon['challenge_rating'] = mon['challenge_rating'].apply(cr_converter)
```

Then I needed to take care of any missing values that I had. This was very easy,
since all of my missing values were related to stats that a number indicated a 
bonus to a die roll. Such as Strength Saving Throw +8 would be recorded as 8.0.
So all I had to do to fill them was fill them with zeros.

```python
mon.fillna(value = 0, inplace = True)
```

After filling all the NaNs, I droped the name column cause I realized it didn't
provide any information, I don't even know why I initially kept it. Then I got
dummy variables for each of my categorical variables. And last but not least,
I made the challenge_rating column the last column in the dataframe.

```python
#Drop the Name column since it doesn't add information
mon = mon.drop('name', axis = 1)

#Get dummy variables for the rest of the categorical variables.
mon = pd.get_dummies(mon, columns = ['size', 'type', 'alignment'],
                        drop_first = True)

#Make challenge_rating the last column
mon = mon[[col for col in mon if col not in ['challenge_rating']]
                            + ['challenge_rating']]
```
Now that the data is clean, I have ploted a few of a features again our target
to get a better idea how they are related. This also helped me get an idea of 
what variables were the most valuable for our model. 

6. Analyze the data with multiple machine learning approaches
7. Evaluate each model
8. Answer the original question
9. Understand and explain potential sources of bias in how your data/model
   answers your question of interest
