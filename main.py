from fastapi import FastAPI, Query, Path
from schema import InputData
from sklearn_inference import sklearn_inference
from onnxinference import onnx_inference
app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}


# @app.post("/predictsklearn/")
# async def predict(input_data: InputData):
#     pred = sklearn_inference(input_data)
#     return {"prediction": pred}

@app.post("/predict")
async def predict(data: InputData,
                  model_type: str = Query("pkl", description="onx or pkl?")):
    
    if(model_type=="pkl"):
        pred = sklearn_inference(data)
        return {"prediction": pred}
    if(model_type=="onx"):
        pred = onnx_inference(data)
        return {"prediction": pred}
    
    return {"error": "model_type must be 'pkl' or 'onx'"}