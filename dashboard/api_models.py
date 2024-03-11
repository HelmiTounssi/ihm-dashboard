from pydantic import BaseModel
from typing import Optional
from typing import List
from enum import Enum
from typing import ClassVar


# Define a Pydantic data model for the response
class ClientIDsResponse(BaseModel):
    client_ids: List[int]


# Define a Pydantic data model for the response
class ClientData(BaseModel):
    id: int
    # Add more fields as needed


# Define a Pydantic data model for error response
class ErrorResponse(BaseModel):
    error: str


class PredictQueryParams(BaseModel):
    # Define the fields for query parameters
    param1: str
    param2: int
    # Add more fields as needed


class ModelName(str, Enum):
    lightgbm: ClassVar[str] = "lightgbm"
    logistic_regression: str = "logistic_regression"


# Define a Pydantic data model for the response
class ClientPredictResponse(BaseModel):
    id: str
    y_pred_proba: float
    y_pred: str
    shap_values: list
    expected_value: float
    client_data: dict


class ClientPredictResponse(BaseModel):
    """Model returned from request to predict client risk"""

    id: int
    y_pred_proba: float
    y_pred: int
    model_type_: Optional[str]
    client_data: Optional[dict]


class ClientExplainResponse(BaseModel):
    """Model returned from request to explain model prediction"""

    id: int
    shap_values: dict
    expected_value: float
    # if return_data==True
    client_data: Optional[dict]
    # if predict==True
    y_pred_proba: Optional[float]
    y_pred: Optional[int]
