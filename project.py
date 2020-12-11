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

#Remove NaN Values
mon1.isnull().sum()

#Because all of the missing values are for stats that simply add to the roll.
# I'm filling them with 0, if a monster gets no bonus to it's Str. save then
# it gets +0

mon1.fillna(value = 0, inplace = True)

#Drop the Name column since it doesn't add information
mon1 = mon1.drop('name', axis = 1)

#Get dummy variables for the rest of the categorical variables.
mon1 = pd.get_dummies(mon1, columns = ['size', 'type', 'alignment'],
                        drop_first = True)

#Make challenge_rating the last column
mon1 = mon1[[col for col in mon1 if col not in ['challenge_rating']]
                            + ['challenge_rating']]


#Run a Random Forest model and see how it does.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

features = mon1.columns[:-1]
target = mon1.columns[-1]

X = mon1[features]
y = mon1[target]

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

#Let's try a Support Vector Machine does.
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

