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
into the code! I first import the modules I'll need then read in the data:

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

3. Clean / wrangle your data
4. Create features
5. Explore the data through EDA
6. Analyze the data with multiple machine learning approaches
7. Evaluate each model
8. Answer the original question
9. Understand and explain potential sources of bias in how your data/model
   answers your question of interest
