from fastapi import FastAPI, HTTPException, Request, Depends
from pydantic import BaseModel
from accelerate import Accelerator
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = FastAPI()


class TextInputs(BaseModel):
    text: str


async def inference(text: str):
    return


@app.post("/predict")
async def predict(text_inputs: TextInputs):
    # 추론 수행
    result = await inference(text_inputs.text)

    # 결과 반환
    return {"predictions": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
