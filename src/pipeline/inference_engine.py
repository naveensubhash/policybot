from typing import List
import pandas as pd
from src.schemas import PolicyInferenceResult
from src.utils import generate_run_id, hash_text, current_timestamp
from src.config import (
    PIPELINE_VERSION,
    AGGREGATION_VERSION,
    REFERENCE_HCPCS_VERSION,
    CONFIDENCE_THRESHOLD
)
from src.pipeline.aggregator import aggregate


class InferenceEngine:
    """
    Core inference orchestration engine.
    
    Coordinates multiple inference methods, aggregates their results,
    and produces structured output with full audit trail.
    """

    def __init__(self, methods: List, hcpcs_df: pd.DataFrame):
        """
        Initialize engine with inference methods.
        
        Args:
            methods: List of InferenceMethod instances to use
            hcpcs_df: HCPCS reference dataframe for descriptions
        """
        self.methods = methods
        self.hcpcs_df = hcpcs_df

    def run(self, policy_text: str) -> PolicyInferenceResult:
        """
        Run inference pipeline on policy text.
        
        Process:
        1. Run all inference methods on the text
        2. Collect evidence from each method
        3. Aggregate evidence into final codes (above threshold only)
        4. Generate audit metadata
        
        Args:
            policy_text: The policy document text to analyze
            
        Returns:
            PolicyInferenceResult with inferred codes and metadata
        """
        # Collect evidence from all methods
        all_evidence = []
        
        for method in self.methods:
            evidence = method.infer(policy_text)
            all_evidence.extend(evidence)
        
        # Aggregate evidence into final codes (filters by threshold)
        final_codes = aggregate(all_evidence, self.hcpcs_df)
        
        # Generate metadata for audit/reproducibility
        metadata = {
            "pipeline_version": PIPELINE_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "run_id": generate_run_id(),
            "input_hash": hash_text(policy_text),
            "reference_hcpcs_version": REFERENCE_HCPCS_VERSION,
            "timestamp": current_timestamp(),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "methods_used": [
                {
                    "method_name": method.method_name,
                    "method_version": method.method_version
                }
                for method in self.methods
            ],
            "num_evidence_collected": len(all_evidence),
            "num_codes_above_threshold": len(final_codes)
        }
        
        return PolicyInferenceResult(
            inferred_codes=final_codes,
            metadata=metadata
        )