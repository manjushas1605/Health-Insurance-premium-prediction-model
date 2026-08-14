from pydantic import BaseModel,Field
import joblib
model=joblib.load("model/insurance_model.pkl")



from fastapi import FastAPI,HTTPException
app=FastAPI(title='Health Insurance Premium Prediction API', 
            description='This API predicts health insurance premiums based on user input.', 
            version='1.0.')

@app.get('/')
def home():
    return{
        "message": "Welcome to the Health Insurance Premium Prediction API!"
    }

#input schema
from pydantic import BaseModel,Field
from typing import Literal

class InsuranceInput(BaseModel):
    age:int=Field(...,ge=18,le=100)
    sex:Literal['male','female']
    bmi:float=Field(...,ge=10,le=60)
    children:int=Field(...,ge=0,le=10)
    smoker:Literal['yes','no']
    region:Literal['southwest','southeast','northwest','northeast']

import pandas as pd 

@app.post("/predict")
def predict_premium(data: InsuranceInput):

    try:

        input_data = pd.DataFrame([
            {
                "age": data.age,
                "sex": data.sex,
                "bmi": data.bmi,
                "children": data.children,
                "smoker": data.smoker,
                "region": data.region
            }
        ])

        prediction = model.predict(input_data)

        return {
            "status": "success",
            "message": "Premium predicted successfully",
            "predicted_premium": round(float(prediction[0]), 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )