import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data_train.csv')
X_train = np.array(df[['roundNumber','half','attackingTeamId','team1Load','team2Load']])
y_train = np.array(df[['winningTeamId']])

df = pd.read_csv('data_test.csv')
X_test = np.array(df[['roundNumber','half','attackingTeamId','team1Load','team2Load']])
y_test = np.array(df[['winningTeamId']])

model = RandomForestClassifier(n_estimators = 100)
model.fit(X_train,y_train)
print(model)

y_pred = model.predict(X_test)
print(y_pred)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test,predictions)
print(accuracy)

#Accuracy = 57.2%