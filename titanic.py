import pandas as pd
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

# Add feature for total family size
df['Family_Size'] = df['SibSp'] + df['Parch']

# Make sex numeric
df['Sex'][df['Sex'] == 'male'] = 0
df['Sex'][df['Sex'] == 'female'] = 1

# Make embark numeric


