"""Type definitions for the Spamlyser application."""
from typing import TypeVar, Dict, Any, List, Union, Optional
from typing_extensions import TypedDict, Protocol

# Type for model predictions
class PredictionDict(TypedDict):
    label: str
    score: float
    spam_probability: Optional[float]

# Type for model statistics
class ModelStats(TypedDict):
    spam: int
    ham: int
    total: int

# Type for method statistics
class MethodStats(TypedDict):
    count: int
    spam: int
    confidences: List[float]

# Type for ensemble predictions
class EnsemblePrediction(TypedDict):
    label: str
    confidence: float
    method: str
    spam_probability: float
    details: str

# Type for model options
class ModelOption(TypedDict):
    id: str
    description: str
    icon: str
    color: str

# Protocol for models
class Model(Protocol):
    def __call__(self, text: Union[str, List[str]]) -> List[PredictionDict]: ...

# Type variables
T = TypeVar('T')
ModelName = str
ModelDict = Dict[ModelName, Model]
OptionsDict = Dict[ModelName, ModelOption]
StatsDict = Dict[ModelName, ModelStats]
MethodStatsDict = Dict[str, MethodStats]

# Type aliases
JsonDict = Dict[str, Any]
Confidence = float
Label = str
