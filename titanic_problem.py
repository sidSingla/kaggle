import pandas as pd

data = pd.read_csv('datasets/titanic_train.csv')
test_data = pd.read_csv('datasets/titanic_test.csv')

df = data.copy()
test_df = test_data.copy()

drop_cols = ['PassengerId', 'Cabin', 'Name', 'Ticket']
df = df.drop(drop_cols, axis=1)
test_df = test_df.drop(drop_cols, axis=1)

one_hot_cols = ['Sex', 'Pclass', 'Embarked']
for col in one_hot_cols:
    df = pd.concat((df,pd.get_dummies(df[col])), axis=1)
    test_df = pd.concat((test_df, pd.get_dummies(test_df[col])), axis=1)

df = df.drop(one_hot_cols,axis=1)
test_df = test_df.drop(one_hot_cols,axis=1)

test_df['Age'] = test_df['Age'].interpolate()
test_df['Fare'] = test_df['Fare'].interpolate()


df = df.dropna()
m, n = df.shape

X = df.drop('Survived',axis=1)
Y = df['Survived']

# Cross Validation
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.1,random_state=0)

from sklearn import tree
max_score = 0
res_depth = 5
for depth in range(5,100):
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train,Y_train)
    score = clf.score(X_test,Y_test)
    if score > max_score:
        max_score = score
        res_depth = depth
        #print (score)
print (max_score, res_depth)

# Fitting data
clf = tree.DecisionTreeClassifier(max_depth=res_depth)

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(base_estimator=clf, n_estimators=100)
model = model.fit(X,Y)

#clf.fit(X,Y)
Y_results = model.predict(test_df)

import numpy as np
matrix = np.column_stack((test_data['PassengerId'], Y_results))
result = pd.DataFrame(matrix,columns=['PassengerId','Survived'])
result.to_csv('predictions.csv', index=False)
