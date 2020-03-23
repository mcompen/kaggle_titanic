import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import string

df = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

"""
columns :   - 'PassengerId', 
            - 'Survived',
            - 'Pclass',
            - 'Name',
            - 'Sex',
            - 'Age',
            - 'SibSp', # siblings + spouses aboard
            - 'Parch', # parents + children aboard
            - 'Ticket', 
            - 'Fare', 
            - 'Cabin', 
            -'Embarked',
"""


def extract_title(name):
    for substr in name.split():
        if substr.endswith("."):
            return substr[0:-1]
    Exception("No title found")


df['Title'] = df['Name'].map(extract_title)
df_test['Title'] = df_test['Name'].map(extract_title)


def all_titles(df):
    title_list = []
    for name in df['Name']:
        title = extract_title(name)
        if title not in title_list:
            title_list.append(title)
    return title_list

# title_list_train = all_titles(df)
# title_list_test = all_titles(df_test)
# all_titles = []
# for title in title_list_train + title_list_test:
#     if title not in all_titles:
#         all_titles.append(title)
# print(all_titles)
# ['Mr', 'Mrs', 'Miss', 'Master', 'Don',
# 'Rev', 'Dr',   'Mme', 'Ms',  'Major',
# 'Lady','Sir', 'Mlle', 'Col',  'Capt',
# 'Countess', 'Jonkheer', 'Dona']


def replace_title(df):  # Replace titles with only Mr, Master, Ms and Miss
    title = df['Title']
    if title in ['Mr', 'Don', 'Rev', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']:
        return 0
    elif title in ['Mrs', 'Lady', 'Mme', 'Countess', 'Dona']:
        return 1
    elif title == 'Master':
        return 2
    elif title in ['Miss', 'Ms', 'Mlle']:
        return 3
    elif title == 'Dr':
        if df['Sex'] == 'Male':
            return 0
        else:
            return 1


# Reduce the variety of titles to only "Mr", "Master", "Ms" and "Miss"
df['Title'] = df.apply(replace_title, axis=1)
df_test['Title'] = df_test.apply(replace_title, axis=1)

# Add feature for total family size
df['Family_Size'] = df['SibSp'] + df['Parch']
df_test['Family_Size'] = df_test['SibSp'] + df_test['Parch']


# Make sex numeric
def replace_sex(df):
    sex = df['Sex']
    if sex == 'male':
        return 0
    else:
        return 1


df['Sex'] = df.apply(replace_sex, axis=1)
df_test['Sex'] = df_test.apply(replace_sex, axis=1)


# Returns all embarks in data set
def all_embarks(df):
    embark_list = []
    for platform in df['Embarked']:
        if platform not in embark_list:
            embark_list.append(platform)
    return embark_list

# print(all_embarks(df))
# ['S', 'C', 'Q', nan]


# Replaces the embarks by numeric values
def replace_embark(df):
    embark = df['Embarked']
    if embark == 'S':
        return 0
    elif embark == 'C':
        return 1
    elif embark == 'Q':
        return 2
    else:  # embarked at unknown place
        return 3


df['Embarked'] = df.apply(replace_embark, axis=1)
df_test['Embarked'] = df_test.apply(replace_embark, axis=1)


# Find all cabin letters
def all_cabinletters(df):
    letter_list = []
    for item, frame in df['Cabin'].iteritems():
        if pd.notnull(frame):
            cabin_letter = df['Cabin'][item][0]
            if cabin_letter not in letter_list:
                letter_list.append(cabin_letter)
    return letter_list


# print(all_cabinletters(df))
# ['C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']

def replace_cabinnumbers(df):
    cnumber = df['Cabin']
    if pd.notnull(cnumber):
        cletter = cnumber[0]
        if cletter == 'C':
            return 0
        elif cletter == 'E':
            return 1
        elif cletter == 'G':
            return 2
        elif cletter == 'D':
            return 3
        elif cletter == 'A':
            return 4
        elif cletter == 'B':
            return 5
        elif cletter == 'F':
            return 6
        elif cletter == 'T':
            return 7
    else:  # unknown cabin
        return 8


df['Cabin'] = df.apply(replace_cabinnumbers, axis=1)
df_test['Cabin'] = df_test.apply(replace_cabinnumbers, axis=1)

# Normalize fares and ages. Use median for missing values
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df['Fare'] = df['Fare'] / df['Fare'].max()
df_test['Fare'] = df_test['Fare'] / df_test['Fare'].max()
df['Age'] = df['Age'].fillna(df['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df['Age'] = df['Age'] / df['Age'].max()
df_test['Age'] = df_test['Age'] / df_test['Age'].max()

# Drop unwanted features
df.drop(columns=['Ticket', 'Name', 'PassengerId'], inplace=True)
test_PId = df_test['PassengerId'].to_numpy()
df_test.drop(columns=['Ticket', 'Name', 'PassengerId'], inplace=True)

y_train = df['Survived'].to_numpy()
X_train = df.drop(columns='Survived', axis=1).to_numpy()
X_test = df_test.to_numpy()

print(df.columns)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

results = pd.DataFrame({
        'PassengerId': test_PId,
        'Survived': y_pred
    })


#Any files you save will be available in the output tab below
results.to_csv('results.csv', index=False)