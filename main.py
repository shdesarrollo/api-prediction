from fastapi import FastAPI
from joblib import load
from fastapi.middleware.cors import CORSMiddleware
import sklearn
import uvicorn

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)

@app.get("/")
async def root():
    return {"message": "Running api"}

@app.post("/predict")
async def predict(data: list):

    model = load('modelo.joblib')
    result = [model.predict([data])]

    return {"result": str(result[0][0][0])} 
    
