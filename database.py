from sqlalchemy import create_engine, Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
import os
from typing import List

DATABASE_URL = os.environ.get('DATABASE_URL')
database = Database(DATABASE_URL)
metadata = MetaData()

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    input_data = Column(String)
    prediction = Column(String)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)

async def save_prediction(input_data: str, prediction: List[float]):
    prediction_result = ' '.join(map(str, prediction))
    query = Prediction.__table__.insert().values(input_data=input_data, prediction=prediction_result)
    await database.execute(query)
