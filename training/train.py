import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data/titanic.csv')

df = df[['Pclass','Sex','Age','Survived']]
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = df.dropna()

X = df[['Pclass','Sex','Age']]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'model.joblib')

print("Model trained successfully")
