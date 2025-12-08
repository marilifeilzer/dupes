from fastapi import FastAPI
from dupes.logic import predict_shampoo

app = FastAPI()

@app.get("/")
def index():
    return {"working": True}

@app.get("/recomend")
def get_recomendation(shampoo: str, description: str):

    if shampoo:

        recomendation = predict_shampoo(shampoo)
    else:

        recomendation = predict_shampoo(description)


    return {"recomendation": recomendation}
