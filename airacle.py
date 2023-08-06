from fastapi import FastAPI
from sqlalchemy import create_engine, Column, Integer, Float, MetaData
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
from sklearn.linear_model import LinearRegression
from pydantic import BaseModel
import numpy as np


class InputData(BaseModel):
    input_data: float


# Database setup
DATABASE_URL = "postgresql://localhost/airacle_fast_api"
database = Database(DATABASE_URL)
metadata = MetaData()

# Async SQLAlchemy core model
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(Float)
    prediction = Column(Float)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

# Train a model
model = LinearRegression()
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model.fit(X, y)

# FastAPI instance
app = FastAPI()

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

@app.post("/predict/")
async def predict(arg: InputData):
    input_array = np.array([[arg.input_data]])
    prediction_result = model.predict(input_array)[0]

    # Store in database
    query = Prediction.__table__.insert().values(input_data=arg.input_data, prediction=prediction_result)
    await database.execute(query)

    return {"prediction": prediction_result}
