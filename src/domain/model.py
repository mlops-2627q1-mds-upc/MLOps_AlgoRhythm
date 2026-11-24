"""Metadata and inference behavior for a machine-learning model."""

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Model(BaseModel):
    """A registered model descriptor that delegates inference to an estimator."""

    model_id: UUID = Field(default_factory=uuid4)
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def get_id(self) -> str:
        """Return the model identifier in the string form used by the UML."""
        return str(self.model_id)

    def get_name(self) -> str:
        """Return the model name."""
        return self.name

    def get_params(self) -> dict[str, Any]:
        """Return the model's recorded parameters."""
        return self.params

    def get_model_info(self) -> dict[str, Any]:
        """Return the model metadata as a JSON-compatible mapping."""
        return {
            "model_id": self.get_id(),
            "name": self.get_name(),
            "params": self.get_params(),
        }

    def predict(self, estimator: Any, features: Any) -> dict[str, Any]:
        """Run inference with a supplied estimator and return its predictions.

        The caller supplies both the loaded estimator and model-ready feature rows,
        keeping this domain object independent of model files and API lifecycle.
        """
        raw_predictions = estimator.predict(features)
        predictions = (
            raw_predictions.tolist()
            if hasattr(raw_predictions, "tolist")
            else list(raw_predictions)
        )
        return {
            "model_id": self.get_id(),
            "model_name": self.get_name(),
            "predictions": predictions,
        }
