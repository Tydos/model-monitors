import pickle
import numpy as np
from schema import InputData
def sklearn_inference(data:InputData) -> float:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)  
    
    # Convert Pydantic → list[float] → numpy array
    X = np.array(data.features, dtype=np.float32).reshape(1, -1)    
    res = model.predict(X)
    return float(res[0])