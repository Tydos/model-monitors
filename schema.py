from pydantic import BaseModel, Field, field_validator

len_features = 12
class InputData(BaseModel):
    features: list[float] = Field(...)

    @field_validator("features")
    def validate_features(cls, features):
        if not features:
            raise ValueError("Input features list cannot be empty.")

        if len(features) != len_features:
            raise ValueError(f"Input features must be a list of {len_features} floats.")

        if len(features) == 0:
            raise ValueError("Input features list cannot be empty.")
        
        return features