from typing import List
from src.schemas import PolicyInferenceResult, Evidence, FinalCode
from src.utils import generate_run_id, hash_text, current_timestamp
from src.config import (
    PIPELINE_VERSION,
    AGGREGATION_VERSION,
    REFERENCE_HCPCS_VERSION
)
from src.pipeline.aggregator import aggregate


class InferenceEngine:
    """
    Core inference orchestration engine.
    
    Coordinates multiple inference methods, aggregates their results,
    and produces structured output with full audit trail.
    """

    def __init__(self, methods: List):
        """
        Initialize engine with inference methods.
        
        Args:
            methods: List of InferenceMethod instances to use
        """
        self.methods = methods

    def run(self, policy_text: str) -> PolicyInferenceResult:
        """
        Run inference pipeline on policy text.
        
        Process:
        1. Run all inference methods on the text
        2. Collect evidence from each method
        3. Aggregate evidence into final codes
        4. Generate audit metadata
        
        Args:
            policy_text: The policy document text to analyze
            
        Returns:
            PolicyInferenceResult with inferred codes, evidence, and audit trail
        """
        # Collect evidence from all methods
        all_evidence = []
        
        for method in self.methods:
            evidence = method.infer(policy_text)
            all_evidence.extend(evidence)
        
        # Aggregate evidence into final codes
        final_codes = aggregate(all_evidence)
        
        # Generate audit metadata for reproducibility
        audit = {
            "pipeline_version": PIPELINE_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "run_id": generate_run_id(),
            "input_hash": hash_text(policy_text),
            "reference_hcpcs_version": REFERENCE_HCPCS_VERSION,
            "timestamp": current_timestamp(),
            "methods_used": [
                {
                    "method_name": method.method_name,
                    "method_version": method.method_version
                }
                for method in self.methods
            ],
            "num_evidence_collected": len(all_evidence),
            "num_codes_inferred": len(final_codes),
            "num_codes_selected": sum(1 for code in final_codes if code.selected)
        }
        
        return PolicyInferenceResult(
            inferred_codes=final_codes,
            evidence=all_evidence,
            audit=audit
        )