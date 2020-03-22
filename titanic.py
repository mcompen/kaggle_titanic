import pandas as pd
import numpy as np
import string

df = pd.read_csv("data/train.csv")
"""
columns :   - 'PassengerId', 
            - 'Survived',
            - 'Pclass',
            - 'Name',
            - 'Sex',
            - 'Age',
            - 'SibSp', # siblings +spouses aboard
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
    return np.nan


def all_titles(df):
    title_list = []
    for name in df['Name']:
        title = extract_title(name)
        if title not in title_list:
            title_list.append(title)
    return title_list


df['Title'] = df['Name'].map(extract_title)

# title_list = all_titles(df)
# print(title_list)
# ['Mr', 'Mrs', 'Miss', 'Master', 'Don',
# 'Rev', 'Dr',   'Mme', 'Ms',  'Major',
# 'Lady','Sir', 'Mlle', 'Col',  'Capt',
# 'Countess', 'Jonkheer']


def replace_title(df):  # Only Mr, Master, Ms and Miss
    title = df['Title']
    if title in ['Don', 'Rev', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']:
        return 'Mr'
    elif title in ['Lady', 'Mme', 'Countess']:
        return 'Mrs'
    elif title in ['Ms', 'Mlle']:
        return 'Miss'
    elif title == 'Dr':
        if df['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


df['Title'] = df.apply(replace_title, axis=1)
df['Family_Size']=df['SibSp']+df['Parch']
