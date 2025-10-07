from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.ai.pipeline import predict_disposition

app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a CSV file and returns prediction probability as JSON.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    try:
        # pass file path or file-like object to your prediction pipeline
        probability = predict_disposition(file.file)

        # If predict_disposition returns a raw float, wrap it in a dict
        if isinstance(probability, (float, int)):
            return {"probability": float(probability)}

        # If it already returns a dict with the probability, just return that
        if isinstance(probability, dict) and "probability" in probability:
            return probability

        # Fallback — ensure a consistent response format
        return {"probability": None, "raw_result": str(probability)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")
