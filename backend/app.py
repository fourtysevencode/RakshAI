from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"status":"up"}

@app.post("/predict")
def predict():
    return {}