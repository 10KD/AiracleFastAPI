from fastapi import FastAPI
from pydantic import BaseModel
from database import database, save_prediction
from model import predict_delay

class InputData(BaseModel):
    input_data: str

app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/predict/")
async def predict(arg: InputData):
    prediction_result = predict_delay(arg.input_data)

    # Store in database
    await save_prediction(arg.input_data, prediction_result)

    return {"prediction": prediction_result}
