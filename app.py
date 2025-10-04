from fastapi import FastAPI

from backend import run_preprocess
from backend.run_preprocess import RunPreprocess
app = FastAPI()

@app.get("/")
async def index():
   return {"message": "Hello World"}

@app.get("/detect")
async def detect():
    run_preprocess = RunPreprocess()
    return run_preprocess.detect()