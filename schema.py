from pydantic import BaseModel, Field

class InputData(BaseModel):
    features: list[float] = Field(...)
  