from fastapi import FastAPI
from detect import detect_router


app=FastAPI()
app.include_router(detect_router)

@app.get('/')
def welcome():
    return{'message':"Welcome to Emotion Recognition model for gaming"}