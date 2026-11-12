from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class Model(BaseModel):
    model_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the model")
    name: str = Field(..., description="Name of the model")
    params: dict = Field(default_factory=dict, description="Parameters of the model")

@property
def get_model_info(self) -> dict:
    """
    Returns a dictionary containing the model's information.
    """
    return {
        "model_id": str(self.model_id),
        "name": self.name,
        "params": self.params
    }

def get_id(self) -> str:
    """
    Returns the unique identifier of the model as a string.
    """
    return str(self.model_id)

def get_name(self) -> str:
    """
    Returns the name of the model.
    """
    return self.name

def get_params(self) -> dict:
    """
    Returns the parameters of the model.
    """
    return self.params

def predict(self) -> dict:
    return {}