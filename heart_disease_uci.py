import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
import pdb
import matplotlib

df = pd.read_csv('datasets/heart.csv')
df = df.dropna()
m, n = df.shape

X = df[df.columns[0:n-1 ]]
Y = df[df.columns[ n-1 ]]

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.5,random_state=0)

from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)
print(score)

model = BaggingClassifier(base_estimator=clf, n_estimators=100)
model = model.fit(X_train, y_train)
score2 = model.score(X_test, y_test)
print (score2)