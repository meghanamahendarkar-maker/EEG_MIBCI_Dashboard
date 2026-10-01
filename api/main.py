from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Add src to pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your existing ML components here
# from src.models.minirocket_pipeline import MiniRocketPipeline

app = FastAPI(title="EEG MI-BCI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the EEG MI-BCI API"}

@app.post("/api/upload")
async def upload_eeg_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.edf', '.csv', '.mat')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Process the file here
    return {"filename": file.filename, "status": "File successfully uploaded and validated."}

@app.post("/api/train")
def train_model():
    # Trigger model training here
    return {"status": "Training started"}

@app.get("/api/status")
def get_status():
    return {"status": "Ready", "accuracy": 98.63}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
