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

```python
#Explore the Data a Bit with various scatter plots
#This will help me get a sense of what will help the model the most.
plot1 = mon.plot(x='hit_points', y='challenge_rating', style='o')
plot2 = mon.plot(x='armor_class', y='challenge_rating', style='o')
plot3 = mon.plot(x='strength', y='challenge_rating', style='o')
plot4 = mon.plot(x='dexterity', y='challenge_rating', style='o')
plot5 = mon.plot(x='constitution', y='challenge_rating', style='o')
plot6 = mon.plot(x='constitution_save', y='challenge_rating', style='o')

#Let's see how the Features we Created Will Help
plot7 = mon.plot(x='num_actions', y='challenge_rating', style='o')
plot8 = mon.plot(x='num_reactions', y='challenge_rating', style='o')
plot9 = mon.plot(x='num_legendary_actions', y='challenge_rating', style='o')
plot10 = mon.plot(x='num_special_abilities', y='challenge_rating', style='o')
plot11 = mon.plot(x='num_spells', y='challenge_rating', style='o')
```

![Figure_1](https://github.com/nhphillips5/DND_Monsters/blob/main/graphs/Figure_1.png)

![Figure_2](https://github.com/nhphillips5/DND_Monsters/blob/main/graphs/Figure_2.png)

Hit points and Armor Class was what I figured would be the best indicators.
And as I suspected they were. The other graphs were interesting as well.
Other than num_actions, the rest of my created features probably didn't add much.
You can view the rest of the graphs in the graphs folder above.

Let's look at all the various classification models I used on this data.
I started out with a Random Forest model. I started with this kind of model 
because it was the first model I learned. This model performed very well with an
f1 score of 98.7% and a macro average recall score of 99.0% I was very impressed.

```python
#Run a Random Forest model and see how it does.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = mon.columns[:-1]
target = mon.columns[-1]

X = mon[features]
y = mon[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

rf = RandomForestClassifier()
rf = rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred = rf.predict(X_test)
y_prob_train = rf.predict_proba(X_train)
y_prob = rf.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))
```
Next I tried a Naive Bays Classifier. I thought this would work well, because
I have had good results with it in the past in natural language processing.
But it only had a f1 score on 36.6% and a macro average recall score of 33.1%.
Pretty pitiful compared to the Random Forest.

```python
#Let's try a Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb = nb.fit(X_train, y_train)

y_pred_train = nb.predict(X_train)
y_pred = nb.predict(X_test)
y_prob_train = nb.predict_proba(X_train)
y_prob = nb.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))
```

The next model I ran was a Logistic Regression model. This model performed
moderately well. with an f1 score of 48.8% and a macro average recall score of
63.3% it was pretty middle of the road.

```python
#Let's try Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr = lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred = lr.predict(X_test)
y_prob_train = lr.predict_proba(X_train)
y_prob = lr.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))

```

Next was a K Nearest Neighbors model. I first ran it with a number of neighbors
equal to the square root of the number of observations in the training dataset.
I did this because I heard it was a good rule of thumb. But the results were
worse that I expected them to be. So I tried running it again with the default
of 5 neighbors and got much better results (though still not great). It had an
f1 score of 47.6% and a macro average recall score of 35.9%.

```python
#Let's see how K Nearest Neighbors does
from sklearn.neighbors import KNeighborsClassifier

#knn = KNeighborsClassifier(n_neighbors = int(math.sqrt(len(X_train))))
#This one didn't do well, let's see out it does with default neighbors.
knn = KNeighborsClassifier()
knn = knn.fit(X_train, y_train)

y_pred_train = knn.predict(X_train)
y_pred = knn.predict(X_test)
y_prob_train = knn.predict_proba(X_train)
y_prob = knn.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))
```

Next I ran a Support Vector Classifier. This model did really well. Not as well
as Random Forest did but still really well. it had an f1 score of 84.7% and a
macro average recall score of 91.4%.

```python
#Let's try a Support Vector Machine.
from sklearn.svm import SVC

svm = SVC(probability = True)
svm = svm.fit(X_train, y_train)

y_pred_train = svm.predict(X_train)
y_pred = svm.predict(X_test)
y_prob_train = svm.predict_proba(X_train)
y_prob = svm.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))
```

And Last but not least, I tried a Gradient Boosted Classifier. I almost didn't 
run this model, but boy am I glad I did! With an f1 score of 99.5% and a macro
average recall score of 99.8% this model clearly takes the cake! I had heard
the stories of how great gradient boosting was, but here I got to see it's 
power first hand. It did take much longer to run than the other models, but 
because I'm not working with gigs of data it wasn't a big deal.

```python
#Let's try one last model Gradient Boosted Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier()
gb = gb.fit(X_train, y_train)

y_pred_train = gb.predict(X_train)
y_pred = gb.predict(X_test)
y_prob_train = gb.predict_proba(X_train)
y_prob = gb.predict_proba(X_test)

print(classification_report(y_train, y_pred_train))
print(recall_score(y_train, y_pred_train, average='macro'))
print(recall_score(y_train, y_pred_train, average='micro'))
print(recall_score(y_train, y_pred_train, average=None))
print(f1_score(y_train, y_pred_train, average='weighted'))
```

So, to answer my original question, machine learning can predict the challenge
rating of monsters in DnD. Not only can it do it, but it can do it well! I'm
excited to use these models to help me get a better idea of how my monsters 
compare with those created by professional DnD R & D teams.

There are certainly things I can try to improve with this model though. There
is a potential source of bias from the data itself. The data is collected from
three sources, the DnD SRD, and the books *Tome of Foes* and *Creature Codex*.
The two books are 3rd party sources of monsters and may not be as accurately 
labeled as official DnD monsters. I would like to get more official data to test
my models.

Another thing I'd like to work with in creating more features. Getting things
like number of attacks and damage resistances I think will improve the model
greatly. I also want to spend some time tuning the hyper parameters of the 
various models. I think it would be neat if I could crack the code and get a 
100% model that isn't overfit. Speaking of overfitting, I want to take a closer
look and make sure my model isn't overfit.

I had a blast working with this data and will continue to tweak and improve it.
Let me know of any improvements you would make or things you would do differently.
