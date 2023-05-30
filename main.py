from fastapi import FastAPI
from joblib import load
import sklearn
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Running api"}

@app.post("/predict")
async def predict(data: list):

    model = load('modelo.joblib')
    result = [model.predict([data])]

    return {"result": str(result[0][0][0])} 
    
