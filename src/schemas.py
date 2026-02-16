from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Evidence:
    code: str
    code_type: str
    method_name: str
    method_version: str
    raw_output: str
    normalized_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    decision_trace: Optional[List[str]] = None  # ← NEW


@dataclass
class FinalCode:
    code: str
    code_description: str
    aggregated_confidence: float
    justification: str
    provenance: Dict[str, Any] = field(default_factory=dict)
    decision_trace: Optional[List[str]] = None  # ← NEW


@dataclass
class PolicyInferenceResult:
    inferred_codes: List[FinalCode]
    metadata: Dict[str, Any]