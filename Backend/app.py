import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import pandas as pd
from model.model import segmentation_model  
from ai_utils import generate_strategy

app = FastAPI()

# Input data model for predictions
class UserInput(BaseModel):
    count_orders: int
    average_spend: float
    return_ratio: float

# Input data model for AI analysis
class CustomerInfo(BaseModel):
    cust_info: str

class FullUserInput(BaseModel):
    age: int
    gender: str
    income: float
    spending_score: float
    membership_years: int
    purchase_frequency: int
    preferred_category: str
    last_purchase_amount: float

# Load model artifacts
kmeans = segmentation_model["kmeans"]
scaler = segmentation_model["scaler"]
le = segmentation_model["le"]
columns = segmentation_model["columns"]
segment_names = segmentation_model["segment_names"]

def predict_segment(input_dict):
    import pandas as pd
    # Create DataFrame
    input_df = pd.DataFrame([input_dict])
    input_df['gender_encoded'] = le.transform(input_df['gender'])
    input_df = pd.get_dummies(input_df, columns=['preferred_category'], prefix='', prefix_sep='')
    input_df.drop('gender', axis=1, inplace=True)
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[columns]
    input_scaled = scaler.transform(input_df)
    segment_label = kmeans.predict(input_scaled)[0]
    segment_name = segment_names.get(segment_label, f"Segment {segment_label}")
    return segment_label, segment_name

@app.get("/")
def home() -> dict:
    """
    Home endpoint to verify service health.
    """
    return {"message": "Welcome to the Customer Segmentation and Strategy Application"}

@app.post("/predict")
def predict_user_segment(data: UserInput) -> dict:
    """
    Predict the user segment based on input data.
    """
    try:
        # Prepare input data for the model
        input_data = pd.DataFrame([data.dict()])
        
        # Ensure the model handles the data appropriately
        input_scaled = scaler.transform(input_data)
        segment_prediction = kmeans.predict(input_scaled)
        if not segment_prediction.any():
            raise ValueError("No prediction received from the model.")

        # Extract and return the result
        segment = int(segment_prediction[0])
        return {"predicted_segment": segment}
    except ValueError as value_error:
        raise HTTPException(status_code=400, detail=str(value_error))
    except Exception as error:
        print(f"Error during prediction: {error}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/ai")
async def get_strategy(data: CustomerInfo):
    # Convert string back to dict if you want structured input
    # Or just pass the raw string into your prompt
    try:
        input_lines = data.cust_info.split("\n")
        input_data = {}
        for line in input_lines:
            if ":" in line:
                k, v = line.split(":", 1)
                input_data[k.strip()] = v.strip()

        strategy = generate_strategy(input_data)
        return {"strategy": strategy}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_segment")
def predict_segment_api(data: FullUserInput) -> dict:
    """
    Predict the user segment using the extended user profile data.
    """
    try:
        input_dict = data.dict()
        print("Received input:", input_dict)
        label, name = predict_segment(input_dict)
        print("Prediction result:", label, name)
        # Convert label to Python int
        return {"segment_label": int(label), "segment_name": str(name)}
    except Exception as error:
        print(f"Error during prediction: {error}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
