import onnxruntime as ort
from schema import InputData
import numpy as np
def onnx_inference(data:InputData) -> float:
    session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Convert to NumPy array and ensure 2D
    X = np.array(data.features, dtype=np.float32).reshape(1, -1)

    # Run prediction
    pred = session.run(None, {input_name: X})
    return float(pred[0][0])