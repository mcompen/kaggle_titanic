import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


def features_to_onehot(df, feature, all_values):
    one_hot = np.zeros((len(df), len(all_values)))
    feature_to_index = {}
    for (ind, val) in enumerate(all_values):
        feature_to_index[val] = ind
    for (ind, val) in enumerate(df[feature]):
        if not pd.isna(val):
            one_hot[ind, feature_to_index[val]] = 1
    for val in all_values:
        df[val] = one_hot[:, feature_to_index[val]]

    df.drop(columns=feature, inplace=True)



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
    if title in ['Don', 'Rev', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']:
        return 'Mr'
    elif title in ['Mrs', 'Lady', 'Mme', 'Countess', 'Dona']:
        return 'Ms'
    elif title == 'Master':
        return 'Master'
    elif title in ['Ms', 'Mlle']:
        return 'Miss'
    elif title == 'Dr':
        if df['Sex'] == 'Male':
            return 'Mr'
        else:
            return 'Ms'


# Reduce the variety of titles to only "Mr", "Master", "Ms" and "Miss"
remaining_titles = ['Mr', 'Master', 'Ms', 'Miss']
df['Title'] = df.apply(replace_title, axis=1)
df_test['Title'] = df_test.apply(replace_title, axis=1)
features_to_onehot(df, 'Title', remaining_titles)
features_to_onehot(df_test, 'Title', remaining_titles)


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


# Find all cabin letters
def all_cabinletters(df):
    letter_list = []
    for item, frame in df['Cabin'].iteritems():
        if pd.notnull(frame):
            cabin_letter = df['Cabin'][item][0]
            if cabin_letter not in letter_list:
                letter_list.append(cabin_letter)
    return letter_list


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
df.drop(columns='Embarked', inplace=True)
df_test.drop(columns='Embarked', inplace=True)

df.drop(columns='Cabin', inplace=True)
df_test.drop(columns='Cabin', inplace=True)

df.drop(columns=['Ticket', 'Name', 'PassengerId'], inplace=True)
df_test.drop(columns=['Ticket', 'Name', 'PassengerId'], inplace=True)

y_train = df['Survived'].to_numpy()[1:500]
X_train = df.drop(columns='Survived', axis=1).to_numpy()[1:500, :]
y_mytest = df['Survived'].to_numpy()[501:890]
X_mytest = df.drop(columns='Survived', axis=1).to_numpy()[501:890, :]

X_test = df_test.to_numpy()

# Logistic Regression
logreg = LogisticRegression(solver="lbfgs", tol=1e-6)
logreg.fit(X_train, y_train)
print("Training accuracy logreg: {}".format(logreg.score(X_train, y_train)))
print("Test accuracy logreg : {}".format(logreg.score(X_mytest, y_mytest)))

# Decision Tree
dectree = DecisionTreeClassifier(max_depth=5)
dectree.fit(X_train, y_train)
from sklearn.tree import export_graphviz
export_graphviz(dectree, out_file="tree.dot", feature_names=df.drop(columns='Survived', axis=1).columns)
print("Training accuracy dectree: {}".format(dectree.score(X_train, y_train)))
print("Test accuracy dectree : {}".format(dectree.score(X_mytest, y_mytest)))

# Random Forest
rforest = RandomForestClassifier(max_depth=5, n_estimators=1000, max_features=2)
rforest.fit(X_train, y_train)
print("Training accuracy rforest: {}".format(rforest.score(X_train, y_train)))
print("Test accuracy rforest : {}".format(rforest.score(X_mytest, y_mytest)))

# Decision tree seems to work best. Now use entire training data
dectree = DecisionTreeClassifier(max_depth=5)
dectree.fit(df.drop(columns='Survived', axis=1), df['Survived'])
y_pred = dectree.predict(X_test)

# Save results to csv
results = pd.DataFrame({'PassengerId': [(p + 892) for p in range(0, len(y_pred))], 'Survived': y_pred})
results.to_csv('results.csv', index=False)

