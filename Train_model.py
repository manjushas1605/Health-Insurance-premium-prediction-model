import pandas as pd
df=pd.read_csv("data/insurance.csv")
print("five sample values:",df.head())
print("column names:",df.columns)
print("describe:",df.describe())
print("find total null values:",df.isnull().sum())
print("shape of dataset:",df.shape)
print("data types of columns:",df.dtypes)

x=df.drop("charges",axis=1)
y=df['charges']





categorical_columns = [
    "sex",
    "smoker",
    "region"
]

numerical_columns = [
    "age",
    "bmi",
    "children"
]

#one hot encoder
#preprocessing
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(handle_unknown='ignore')

#use ColumnTransformer
from sklearn.compose import ColumnTransformer
preprocessor=ColumnTransformer(
    transformers=[("categorical",
encoder,
categorical_columns
)],
remainder='passthrough'
)

print("\n Preprocessing done successfully")

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
from sklearn.pipeline import Pipeline

model=Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("regressor",regressor)
    ]
)

model.fit(x_train,y_train)
print("model trained successfully!")


y_pred=model.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)



import joblib
joblib.dump(model,"insurance_model.pkl")



                       









