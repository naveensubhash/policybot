from typing import List, Dict
import pandas as pd
from src.schemas import Evidence, FinalCode
from src.config import CONFIDENCE_THRESHOLD, REFERENCE_HCPCS_VERSION


def aggregate(evidences: List[Evidence], hcpcs_df: pd.DataFrame) -> List[FinalCode]:
    """
    Aggregate evidence from multiple inference methods.
    
    Strategy: Max confidence aggregation
    - Groups evidence by code
    - Takes maximum confidence across all methods
    - Only includes codes above confidence threshold
    - Builds rich provenance metadata per code
    
    Args:
        evidences: List of Evidence objects from all methods
        hcpcs_df: DataFrame with HCPCS codes and descriptions
        
    Returns:
        List of FinalCode objects with aggregated confidence scores
    """
    # Group evidence by code
    grouped: Dict[str, List[Evidence]] = {}
    
    for ev in evidences:
        grouped.setdefault(ev.code, []).append(ev)
    
    final_codes = []
    
    for code, ev_list in grouped.items():
        # Take maximum confidence across all evidence for this code
        max_conf = max(ev.normalized_confidence for ev in ev_list)
        
        # FILTER: Only include codes above threshold
        if max_conf < CONFIDENCE_THRESHOLD:
            continue
        
        # Get the best (highest confidence) evidence for this code
        best_evidence = max(ev_list, key=lambda e: e.normalized_confidence)
        
        # Get code description from HCPCS reference
        code_row = hcpcs_df[hcpcs_df['code'] == code]
        code_description = ""
        if not code_row.empty:
            code_description = code_row.iloc[0]['description']
        
        # Build provenance metadata
        method_type = "deterministic" if "mock" in best_evidence.method_name else "llm"
        
        provenance = {
            "method": {
                "name": best_evidence.method_name,
                "type": method_type,
                "version": best_evidence.method_version
            },
            "reference_data": {
                "hcpcs_version": REFERENCE_HCPCS_VERSION,
                "hcpcs_description": code_description
            }
        }
        
        # Add model info (works for both mock and LLM methods)
        if best_evidence.metadata.get("api_provider"):
            # Real LLM method
            provenance["model"] = {
                "name": best_evidence.metadata.get("model_name"),
                "provider": best_evidence.metadata.get("api_provider"),
                "tokens_used": best_evidence.metadata.get("total_tokens")
            }
        else:
            # Mock/deterministic method
            provenance["model"] = {
                "name": best_evidence.metadata.get("model_name"),
                "version": best_evidence.metadata.get("model_version"),
                "type": "mock"
            }
        
        # Create FinalCode with decision_trace
        final_codes.append(
            FinalCode(
                code=code,
                code_description=code_description,
                aggregated_confidence=max_conf,
                justification=best_evidence.raw_output,
                provenance=provenance,
                decision_trace=best_evidence.decision_trace  # ← NEW
            )
        )
    
    # Sort by confidence (highest first)
    final_codes.sort(key=lambda x: x.aggregated_confidence, reverse=True)
    
    return final_codes