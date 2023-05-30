from fastapi import FastAPI
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import sklearn
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Running api"}

@app.post("/predict")
async def predict(data: list):

    model = load('modelo.joblib')
    print(model.predict([data]))
    result = [model.predict([data])]
    data = result[0][0]
    print(result[0][0][0])
    print(data[0])

    return {"result": str(result[0][0][0])} 
    