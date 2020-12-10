import numpy as np
import pandas as pd
import re

#Files I will be reading in
filenames = ["monsters/mon1.json",
             "monsters/mon2.json",
             "monsters/mon3.json",
             "monsters/mon4.json",
             "monsters/mon5.json",
             "monsters/mon6.json",
             "monsters/mon7.json",
             "monsters/mon8.json",
             "monsters/mon9.json",
             "monsters/mon10.json",
             "monsters/mon11.json",
             "monsters/mon12.json",
             "monsters/mon13.json",
             "monsters/mon14.json",
             "monsters/mon15.json",
             "monsters/mon16.json",
             "monsters/mon17.json",
             "monsters/mon18.json",
             "monsters/mon19.json",
             "monsters/mon20.json",
             "monsters/mon21.json",
             "monsters/mon22.json"]

#Make a DataFrame
monsters = []
for f in filenames:
    monsters.append(pd.read_json(f))

mon = pd.DataFrame()

mon = mon.append(monsters)

mon = mon['results'].apply(pd.Series)

#Make a dataframe that requires minimal Cleaning, if more time add more features

mon1 = mon.drop(['slug', 'subtype','group', 'armor_desc', 'hit_dice', 'speed',
    'skills', 'damage_vulnerabilities', 'damage_resistances',
    'damage_immunities', 'condition_immunities', 'senses', 'languages',
    'actions', 'reactions', 'legendary_desc','legendary_actions',
    'special_abilities', 'spell_list','img_main', 'document__slug',
    'document__title', 'document__license_url'], axis = 1)

#Convert Challenge rating to integers. make the fractions whole numbers and
#raise the rest accordingly.

mon1.challenge_rating.unique()

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
mon1['challenge_rating'] = mon1['challenge_rating'].apply(cr_converter)

#Check Dataset
mon1.head()
mon1.challenge_rating.unique()



#Run a Random Forest model and see how it does.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#TODO Drop name column and get dummies

features = mon1.columns[:-1]
response = mon1.columns[-1]

X = mon1[features]
y = mon1[response]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

rf = RandomForestClassifier()
rf = rf.fit(X_train, y_train)

