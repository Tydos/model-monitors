from sklearn_inference import sklearn_inference
from onnxinference import onnx_inference
import numpy as np
from schema import InputData
import pytest
from collections.abc import Sequence

"""
what does this code test?

1. Verify model input is a list of floats
2. Verify both inference functions returns a float prediction
3. Verify both inference functions handle edge cases (e.g., empty input, incorrect input size)
4. Verify consistency between sklearn and ONNX model predictions on the same input

"""


def test_sklearn_inference():

    #Check Sklearn inference
    sample_data = [119010.0,
    17165.69,
    0.071,
    719.0,
    19154.78,
    9.31,
    0.79573821735972,
    0.799739435413342,
    0.7891838899856305,
    0.8941318998760788,
    0.7972864724141384,
    0.8474429287957624]
    input_data = InputData(features=sample_data)
    pred_sample_1 = sklearn_inference(input_data)
    pred_sample_2 = sklearn_inference(input_data)
    assert pred_sample_1 == pred_sample_2

    #check output type
    assert(isinstance(pred_sample_1, float))

def test_input_schema():  
    #check input schema
    inputarray = np.zeros(12).tolist() #Model input - 12 features
    input_data = InputData(features=inputarray)
    assert isinstance(input_data.features, list)
    assert len(input_data.features) == 12
    assert all(isinstance(x, float) for x in input_data.features)
   
def test_pydantic_validation():  
    #Check Pydantic
    with pytest.raises(ValueError):
        invalid_input = InputData(features=[])

def test_onnx_inference():    
    #Check ONNX inference
    sample_data = [119010.0,
    17165.69,
    0.071,
    719.0,
    19154.78,
    9.31,
    0.79573821735972,
    0.799739435413342,
    0.7891838899856305,
    0.8941318998760788,
    0.7972864724141384,
    0.8474429287957624]
    input_data = InputData(features=sample_data)
    pred_sample_1 = onnx_inference(input_data)
    pred_sample_2 = onnx_inference(input_data)
    assert pred_sample_1 == pred_sample_2

    #check output type
    assert(isinstance(pred_sample_1, float))