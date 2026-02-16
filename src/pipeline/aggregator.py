from typing import List, Dict
from src.schemas import Evidence, FinalCode
from src.config import CONFIDENCE_THRESHOLD


def aggregate(evidences: List[Evidence]) -> List[FinalCode]:
    """
    Aggregate evidence from multiple inference methods.
    
    Strategy: Max confidence aggregation
    - Groups evidence by code
    - Takes maximum confidence across all methods
    - Selects codes meeting confidence threshold
    
    Args:
        evidences: List of Evidence objects from all methods
        
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
        
        # Select if confidence meets threshold
        selected = max_conf >= CONFIDENCE_THRESHOLD
        
        # Use the justification from the highest-confidence evidence
        best_evidence = max(ev_list, key=lambda e: e.normalized_confidence)
        justification = best_evidence.raw_output
        
        final_codes.append(
            FinalCode(
                code=code,
                aggregated_confidence=max_conf,
                selected=selected,
                justification=justification
            )
        )
    
    # Sort by confidence (highest first)
    final_codes.sort(key=lambda x: x.aggregated_confidence, reverse=True)
    
    return final_codes