from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

model = pickle.load(open("model/spam_model.pkl", "rb"))

class Message(BaseModel):
    message: str

@app.get("/")
def home():
    return {"message": "Spam Detection API running"}


@app.post("/predict")
def predict(data: Message):
    
    prediction = model.predict([data.message])[0]

    return {
        "message": data.message,
        "prediction": prediction
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)