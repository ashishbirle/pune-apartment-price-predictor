import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("apartment-price-predictor/data/pune_apartment.csv")
#print(df.head())

le_area = LabelEncoder()
le_yesno = LabelEncoder()

df['area'] = le_area.fit_transform(df['area'])

df['parking'] = le_yesno.fit_transform(df['parking'])
df['balcony'] = le_yesno.fit_transform(df['balcony'])
df['gym'] = le_yesno.fit_transform(df['gym'])
df['security'] = le_yesno.fit_transform(df['security'])
df['swimming_pool'] = le_yesno.fit_transform(df['swimming_pool'])

print(df.head())

X = df.drop('price', axis=1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("The model scored", model.score(X_test, y_test))

with open("apartment-price-predictor/models/model_rf.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully")
