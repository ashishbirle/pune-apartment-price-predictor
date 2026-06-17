import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("apartment-price-predictor\data\pune_apartment.csv")

print(df.head())

le_area = LabelEncoder()
le_yesno = LabelEncoder()
df['area'] = le_area.fit_transform(df['area'])
df['parking'] = le_yesno.fit_transform(df['parking'])
df['balcony'] = le_yesno.fit_transform(df['balcony'])
df['gym'] = le_yesno.fit_transform(df['gym'])
df['security'] = le_yesno.fit_transform(df['security'])
df['swimming_pool'] = le_yesno.fit_transform(df['swimming_pool'])

#print(df.head())

X = df.drop('price', axis=1)
y = df.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()


model.fit(X_train, y_train)
r2_score = model.score(X_test, y_test) 

with open("apartment-price-predictor/models/model_lr.pkl", "wb") as f:
    pickle.dump(model, f)

with open("apartment-price-predictor/models/area_encoder.pkl", "wb") as f:
    pickle.dump(le_area, f)

with open("apartment-price-predictor/models/yesno_encoder.pkl", "wb") as f:
    pickle.dump(le_yesno, f)

print("Model saved successfully")