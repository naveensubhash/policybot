from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Evidence:
    code: str
    code_type: str
    method_name: str
    method_version: str
    raw_output: str
    normalized_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FinalCode:
    code: str
    aggregated_confidence: float
    selected: bool
    justification: str


@dataclass
class PolicyInferenceResult:
    inferred_codes: List[FinalCode]
    evidence: List[Evidence]
    audit: Dict[str, Any]