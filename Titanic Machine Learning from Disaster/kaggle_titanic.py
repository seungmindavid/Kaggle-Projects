# Kaggle Competition
# Titanic: Machine Learning from Disaster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape)
print(train.head())
print(train.dtypes)
print(test.shape)
print(test.dtypes)
print("null in Training Data")
print(train.isnull().sum())
print("null in Testing Data")
print(test.isnull().sum())
# Training:
# 177 data is missing in Age column
# 687 data is missing in Cabin column
# 2 data is missing in Embarked column

# Testing:
# 86 data is missing in Age column
# 1 data is missing in Fare
# 327 data is missing in Cabin column

# Object values:
# Name, Sex, Ticket, Cabin, Embarked

# Let's Convert Sex into numeric value
# Sex is converted!
train['Sex'] = train['Sex'].map({"male":0, "female":1})
test['Sex'] = test['Sex'].map({"male":0, "female":1})
# 0 : Male
# 1 : Female

print(train['Embarked'].unique())

# Let's drop the row with nan value
train = train.dropna(axis=0, subset=['Embarked'])
test = test.dropna(axis=0, subset=['Embarked'])
train['Embarked'].value_counts()
# S: 0
# C: 1
# Q: 2

# Convert to numeric value
# Embarked is converted!
train['Embarked']= train['Embarked'].map({"S":0, "C":1, "Q":2})
test['Embarked']= test['Embarked'].map({"S":0, "C":1, "Q":2})

# Name is converted to Title with numeric values!
train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)
train = train.drop(['Name'], axis=1)
test['Title'] = test['Name'].str.extract('([A-Za-z]+)\.', expand=False)
test = test.drop(['Name'], axis=1)

train['Title'].value_counts()
train['Title'] = train['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2})
test['Title'] = test['Title'].map({'Mr':0, 'Miss':1, 'Mrs':2})
train['Title'].fillna(3.0, inplace=True)
test['Title'].fillna(3.0, inplace=True)
train.head()

# Let's convert Cabin.
print("Train data's Cabin Value Counts")
print(train['Cabin'].value_counts())
# It's named alphabetically. Alphabet might be differentiated by Pclass.
train['Cabin'] = train['Cabin'].str[:1]
test['Cabin'] = test['Cabin'].str[:1]
print(train['Cabin'].value_counts())

train['Cabin'] = train['Cabin'].map({'C':0.2,'B':0.4,'D':0.6,'E':0.8,'A':1.0,'F':1.2,'G':1.4, 'T': 1.6})
test['Cabin'] = test['Cabin'].map({'C':0.2,'B':0.4,'D':0.6,'E':0.8,'A':1.0,'F':1.2,'G':1.4, 'T': 1.6})
#print(train.groupby('Pclass')['Cabin'].agg(lambda x: x.value_counts().index[0]))
#print(test.groupby('Pclass')['Cabin'].agg(lambda x: x.value_counts().index[0]))

# Cabin is converted!



# Let's fill nan value in Age,Cabin and Fare
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'), inplace=True)
test['Age'].fillna(test.groupby('Title')['Age'].transform('median'), inplace=True)
print('Right before cabin fillna')
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'), inplace=True)

# We need to convert fare value into 0~3
train.loc[train['Fare'] <= 17, 'Fare'] = 0
train.loc[(train['Fare'] > 17) & (train['Fare'] <= 30), 'Fare'] = 1
train.loc[(train['Fare'] > 30) & (train['Fare'] <= 100), 'Fare'] = 2
train.loc[train['Fare'] > 100, 'Fare'] = 3

test.loc[test['Fare'] <= 17, 'Fare'] = 0
test.loc[(test['Fare'] > 17) & (test['Fare'] <= 30), 'Fare'] = 1
test.loc[(test['Fare'] > 30) & (test['Fare'] <= 100), 'Fare'] = 2
test.loc[test['Fare'] > 100, 'Fare'] = 3

# There are 88 different ages. Let's divide them into 
# 0: 0 ~ 6 Very young. Need to be take care of
# 1: 6 ~ 19 Children of parents
# 2: 20 ~ 30 energetic
# 3: 30 ~ 60 adult who can take care of themselves and their family
# 4: 60 ~ 100 senior
print('right before age converting')
train.loc[train['Age'] <= 13 ,'Age'] = 0
train.loc[(train['Age'] > 13) & (train['Age'] <= 20), 'Age'] = 1
train.loc[(train['Age'] > 20) & (train['Age'] <= 33), 'Age'] = 2
train.loc[(train['Age'] > 33) & (train['Age'] <= 45), 'Age'] = 3
train.loc[(train['Age'] > 45), 'Age'] = 4

test.loc[test['Age'] <= 13 ,'Age'] = 0
test.loc[(test['Age'] > 13) & (test['Age'] <= 20), 'Age'] = 1
test.loc[(test['Age'] > 20) & (test['Age'] <= 33), 'Age'] = 2
test.loc[(test['Age'] > 33) & (test['Age'] <= 45), 'Age'] = 3
test.loc[(test['Age'] > 45), 'Age'] = 4

#train['Familymember'] = train['SibSp'] + train['Parch']
#train = train.drop(['SibSp', 'Parch'], axis=1)

#test['Familymember'] = test['SibSp'] + test['Parch']
#test = test.drop(['SibSp', 'Parch'], axis=1)
#train.head()

# Ticket name is not useful
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)

# We're done with Feature Engineering.
# Let's apply Min-Max Scaling before Machine Learning
# Reason for Min-Max Scaling is ''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC

#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score


X = train[['Pclass', 'Sex', 'Age', 'Embarked', 'Title', 'Cabin','SibSp', 'Parch','Fare']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state =0)

#scaler = MinMaxScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.fit_transform(X_test)
print('here?')
xgbclf = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
rfclf = RandomForestClassifier().fit(X_train, y_train)
svm = SVC(kernel='rbf').fit(X_train,y_train)

rfclf_train_score = rfclf.score(X_train, y_train)
rfclf_test_score = rfclf.score(X_test, y_test)
svm_train_score = svm.score(X_train, y_train)
svm_test_score = svm.score(X_test, y_test)
xgb_train_score = xgbclf.score(X_train, y_train)
xgb_test_score = xgbclf.score(X_test, y_test)
print(rfclf_train_score)
print(rfclf_test_score)
print(svm_train_score)
print(svm_test_score)
print(xgb_train_score)
print(xgb_test_score)

# I'll use the svm

testX = test[['Pclass', 'Sex', 'Age', 'Embarked', 'Title','Cabin','SibSp', 'Parch', 'Fare']]
#testscaler = MinMaxScaler()
#testX = testscaler.fit_transform(testX)
#trainX = testscaler.fit_transform(X)
#trainX_scaled = testscaler.fit_transform(X)
clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05).fit(X, y)
print(clf.score(X,y))
result = clf.predict(testX)
submission = pd.DataFrame({"PassengerId": test['PassengerId'],
                           "Survived": result})
submission.to_csv('titanic_prediction.csv', index=False)
