from fastapi import FastAPI
import pandas as pd
from dupes.logic import predict_shampoo
from dupes.model.descriptions_chromadb import embedding_description_get_recommendation

app = FastAPI()

@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(shampoo: str, description: str):

    recomendation = embedding_description_get_recommendation(description)

    return recomendation
