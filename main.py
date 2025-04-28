from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI(title="Income Prediction API 🚀")

# Load model pipeline
model = None

@app.on_event("startup")
def load_model():
    global model
    with open("pipeline.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model pipeline loaded successfully!")

# Input schema
class Person(BaseModel):
    age: int
    fnlwgt: int
    educational_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str

# Fungsi bikin age_group
def get_age_group(age):
    if age <= 25:
        return 'Muda'
    elif age <= 45:
        return 'Dewasa'
    elif age <= 65:
        return 'Paruh_baya'
    else:
        return 'Lansia'

# Preprocess input
def preprocess_input(data: Person):
    df = pd.DataFrame([data.dict()])
    
    # Tambah kolom gender_Male
    df['gender_Male'] = df['sex'].map({'Male': 1, 'Female': 0})
    
    # Tambah kolom age_group
    df['age_group'] = df['age'].apply(get_age_group)
    
    # Drop kolom sex
    df.drop(columns=['sex'], inplace=True)
    
    # One-hot encoding manual untuk education
    education_categories = [
        '11th', '12th', '1st_4th', '5th_6th', '7th_8th', '9th',
        'Assoc_acdm', 'Assoc_voc', 'Bachelors', 'Doctorate', 'HS_grad',
        'Masters', 'Preschool', 'Prof_school', 'Some_college'
    ]
    for cat in education_categories:
        df[f'education_{cat}'] = (df['education'] == cat).astype(int)
    df.drop(columns=['education'], inplace=True)
    
    # One-hot encoding manual untuk marital_status
    marital_status_categories = [
        'Married_AF_spouse', 'Married_civ_spouse', 'Married_spouse_absent',
        'Never_married', 'Separated', 'Widowed'
    ]
    for cat in marital_status_categories:
        df[f'marital_status_{cat}'] = (df['marital_status'] == cat).astype(int)
    df.drop(columns=['marital_status'], inplace=True)

    return df

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "Income Prediction API is running 🚀"}

# Endpoint predict
@app.post("/predict")
def predict_income(data: Person):
    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        result = ">50K" if prediction == 1 else "<=50K"
        return {
            "prediction": int(prediction),
            "result": result
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
