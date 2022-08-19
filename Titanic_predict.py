import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import io
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st


st.set_page_config(layout = 'wide')

st.write("""
# Titanic Survival Predict

This app predicts  the ***Survival on the Titanic***!

This problem in on [kaggle](https://www.kaggle.com/competitions/titanic/data?select=train.csv)
""")

st.write('---')

st.write('')

col2, col3 = st.columns((2,1))
# Load data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test
train, test = load_data()
buffer = io.StringIO()


col2.write('### Training Data')
col2.dataframe(train)

col2.write("""
* Survived: Survival (0 = No, 1 = Yes)
* Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
* Sex: Sex
* Age: Age in years
* SibSp: # of siblings / spouses aboard the Titanic
* Parch: # of parents / children aboard the Titanic
* Ticket: Ticket number
* Fare: Passenger fare
* Cabin: Cabin number
* Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

""")
train.info(buf=buffer)
s = buffer.getvalue()

col2.text(s)

col2.write('---')
col2.subheader('Visualization Analysis')
col2.write("""
After visualizing, we can come to some conclusions:
* Females are more likely to have a higher chance of surviving than males. This is true because in real life, women and children are prioritized for evacuation when in a dangerous situation
* First class passengers also have a higher chance of surviving than other classes, due to the fact that first class passengers are prioritized over other classes
* In the heatmap plot we can see that "Fare" has a high correlation with "Survival". This is because of the first class treatment which we mentioned above
""")

col3.subheader('Plot to understand the data\'s relationships ')
## Plot heatmap
df_num = train[['Survived','Age','SibSp','Parch','Fare']]
corr = df_num.corr()
mask = np.ones_like(corr)
mask = np.triu(mask)
mask = mask[1:, :-1]
corr = corr.iloc[1:, :-1]
fig = plt.figure()
sns.heatmap(corr, mask= mask, cmap= 'mako', annot= True, annot_kws=({'fontsize' : 12}))
col3.pyplot(fig)

## Plot 
fig = plt.figure()
splot = sns.pointplot(data= train, x= 'Pclass',  y= 'Survived', hue= 'Sex')
splot.set(ylim = (0, 1.1))
splot.set(title='Pclass Survival')
col3.pyplot(fig)

fig = plt.figure()
splot = sns.barplot(train['Embarked'], train['Survived'])
splot.set(title='Embarked Survival')
col3.pyplot(fig)

fig = plt.figure()
splot = sns.barplot(train['Sex'], train['Survived'])
splot.set(ylim = (0,1))
splot.set(title='Sex Survival')
# for p in splot.patches:
#     splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
col3.pyplot(fig)

fig = plt.figure()
sns.barplot(data= train, x= 'SibSp', y= 'Survived')
plt.title('Sibsp Survival')
col3.pyplot(fig)

fig = plt.figure()
sns.barplot(data= train, x= 'Parch', y= 'Survived')
plt.title('Parch Survival')
col3.pyplot(fig)

fig = plt.figure()
sns.kdeplot(train['Age'][train['Survived'] == 0], label= 'Age did not Survive')
sns.kdeplot(train['Age'][train['Survived'] == 1], label= 'Age Survived')
plt.title('Age Survival')
plt.legend(loc ='best')
col3.pyplot(fig)

fig = plt.figure()
sns.distplot(train['Fare'])
plt.title('Before Fare\'s Histogram')
col3.pyplot(fig)




### Using Tukey Method To Detect Outliers

def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers

Outliers_to_drop = detect_outliers(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])

### Preprocessing

col2.write('---')
col2.subheader('Preprocessing Data')
col2.write("""
Using Tukey method to detect Outliers  
* Outliers Data detected
""")
col2.dataframe(train.loc[Outliers_to_drop, :])
before = len(train)
train = train.drop(Outliers_to_drop)
after = len(train)
### Dropping Outliers
col2.write("""
* Drop all the Outliers Data detected
* Before dropping data, training data have **"""+ str(before) +"""** rows
* After dropping data, training data have **"""+ str(after) +"""** rows
* Because "Fare"'s histogram is skewed to the right, we transform the data by using Log transformation
""")

train.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace= True)
test.drop(['Name', 'Ticket', 'Cabin'], axis = 1, inplace= True)

col2.write("""
Dropping features
* We dropped three features "Name", "Ticket" and "Cabin" on both train and test data
""")

col2.write("""
Fill missing values
* Fill in "Embarked"'s missing values in Training Data with the most occurring values
* Fill in "Fare"'s missing values in Testing Data with its mode because we get the most purchased ticket price
* Fill in "Age"'s missing values in both Testing and Training Data with its mean
""")
train['Embarked'].fillna(train['Embarked'].value_counts().idxmax(), inplace= True)
test['Fare'].fillna(test['Fare'].mode()[0], inplace= True)

train['Age'].fillna(train['Age'].mean(), inplace= True)
test['Age'].fillna(test['Age'].mean(), inplace= True)

train['Fare'] = train['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
test['Fare'] = test['Fare'].map(lambda x: np.log(x) if x > 0 else 0)

fig =  plt.figure()
sns.distplot(train['Fare'])
plt.title('After Fare\' Histogram')
col3.pyplot(fig)

col2.write("""
Encoding
* Change Male and Female to 0 and 1. Apply in both Training and Test Data
* Divide "Age" into groups. Apply in both Training and Test Data
* Divide "Fare" into groups. Apply in both Training and Test Data
* Turn "Embarked" into dummies variables
* Check "SibSp" and "Parch" to see if passengers travel alone. 1 is alone 0 is not
* Dop "PassengerId" column
**Training Data after performing the Preprocessing process**
""")

train['Sex']= train['Sex'].map({'male' : 0, 'female' : 1})
test['Sex']= test['Sex'].map({'male' : 0, 'female' : 1})

train.loc[train['Age']  <= 16.336, 'Age'] = 0
train.loc[(train['Age']  > 16.336) & (train['Age'] <= 32.252), 'Age'] = 1
train.loc[(train['Age']  > 32.252) & (train['Age'] <= 48.168), 'Age'] = 2
train.loc[(train['Age']  > 48.168) & (train['Age'] <= 64.084), 'Age'] = 3
train.loc[(train['Age']  > 64.084), 'Age'] = 4

test.loc[test['Age']  <= 16.336, 'Age'] = 0
test.loc[(test['Age']  > 16.336) & (test['Age'] <= 32.252), 'Age'] = 1
test.loc[(test['Age']  > 32.252) & (test['Age'] <= 48.168), 'Age'] = 2
test.loc[(test['Age']  > 48.168) & (test['Age'] <= 64.084), 'Age'] = 3
test.loc[(test['Age']  > 64.084), 'Age'] = 4



train.loc[train['Fare']  <= 2.08, 'Fare'] = 0
train.loc[(train['Fare']  > 2.08) & (train['Fare'] <= 4.159), 'Fare'] = 1
train.loc[(train['Fare']  > 4.159), 'Fare'] = 2


test.loc[test['Fare']  <= 2.08, 'Fare'] = 0
test.loc[(test['Fare']  > 2.08) & (test['Fare'] <= 4.159), 'Fare'] = 1
test.loc[(test['Fare']  > 4.159), 'Fare'] = 2

train = pd.get_dummies(data= train, columns= ['Embarked'], prefix = 'Embarked')
test = pd.get_dummies(data= test, columns= ['Embarked'], prefix = 'Embarked')

train['Alone'] = train['SibSp'] + train['Parch'] + 1
train['Alone'] = train['Alone'].apply(lambda x: 1 if x == 1 else 0)

test['Alone'] = test['SibSp'] + test['Parch'] + 1
test['Alone'] = test['Alone'].apply(lambda x: 1 if x == 1 else 0)

train.drop(['SibSp', 'Parch'], inplace= True, axis= 1)
test.drop(['SibSp', 'Parch'], inplace= True, axis= 1)

train.drop(['PassengerId'], axis= 1, inplace= True)
test.drop(['PassengerId'], axis= 1, inplace= True)

col2.dataframe(train)
col2.write('**Testing Data after performing the Preprocessing process**')
col2.dataframe(test)


col2.write('---')
col2.subheader('Modeling')

result_og = pd.read_csv('gender_submission.csv')
X_train = train.drop('Survived', axis = 1)
Y_train = train['Survived']
X_test = test.copy()
result = result_og['Survived']


ranforest = RandomForestClassifier()
ranforest.fit(X_train, Y_train)
Y_pred = ranforest.predict(X_test)

ranforest_acc = accuracy_score(Y_pred, result)*100

logres = LogisticRegression()
logres.fit(X_train, Y_train)
Y_pred = logres.predict(X_test)
train_hat = logres.predict(X_train)
logres_acc = accuracy_score(Y_pred, result)*100

compare = pd.DataFrame({'PassengerId' : result_og['PassengerId'],'Predicted': Y_pred})
compare['Actual'] = result

col2.write("""
We will use RandomForestClassifier and LogisticRegression in the sklearn library for this classification problem
* After fitting both models we have
* The RandomForestClassifier accuracy is **"""+ str(ranforest_acc) +"""** 
* The LogisticRegression accuracy is **"""+ str(logres_acc) +"""** 
* We chose LogisticRegression over RandomForestClassifier model because it has better accuracy. Even with high accuracy, it won't overfit because the data is tested on a different dataset
""")
col2.dataframe(compare)