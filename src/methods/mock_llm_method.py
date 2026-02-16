import pandas as pd
from typing import List, Set
from src.methods.base_method import InferenceMethod
from src.schemas import Evidence
from src.config import MODEL_NAME, MODEL_VERSION, PROMPT_TEMPLATE_VERSION


class MockLLMMethod(InferenceMethod):
    """
    Mock LLM inference method using keyword matching.
    """

    def __init__(self, hcpcs_df: pd.DataFrame):
        self.hcpcs_df = hcpcs_df
        self.method_name = "mock_llm_reasoning"
        self.method_version = "1.0"
        
        # Stop words to ignore
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'than', 'too', 'very', 'just', 'only',
            # Medical-specific common words
            'procedure', 'service', 'treatment', 'patient', 'medical'
        }

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        words = text.lower().split()
        keywords = set()
        
        for word in words:
            # Clean punctuation
            clean = word.strip('.,;:!?()[]{}"\'-/')
            
            # Keep if 3+ chars, not stop word, not pure number
            if (len(clean) >= 3 and 
                clean not in self.stop_words and 
                not clean.isdigit()):
                keywords.add(clean)
        
        return keywords

    def infer(self, policy_text: str) -> List[Evidence]:
        """
        Infer HCPCS codes using simple keyword matching.
        
        Rules:
        - Need at least 3 matching keywords
        - Confidence = 0.5 + (0.1 * num_matches), capped at 0.95
        """
        evidences = []
        
        if not policy_text or not policy_text.strip():
            return evidences
        
        # Extract keywords
        policy_keywords = self._extract_keywords(policy_text)
        
        if not policy_keywords:
            return evidences
        
        # Match against each HCPCS code
        for _, row in self.hcpcs_df.iterrows():
            code = str(row['code'])
            description = str(row['description'])
            
            # Extract description keywords
            desc_keywords = self._extract_keywords(description)
            
            # Find matches
            matching = policy_keywords.intersection(desc_keywords)
            num_matches = len(matching)
            
            # Require at least 3 matches
            if num_matches >= 3:
                # Calculate confidence: 0.5 base + 0.1 per match
                confidence = min(0.5 + (num_matches * 0.1), 0.95)
                
                # Format matched terms
                matched_terms = ', '.join(sorted(list(matching))[:5])
                
                reasoning = (
                    f"Found {num_matches} matching keywords ({matched_terms}) "
                    f"between policy text and HCPCS {code}: '{description}'"
                )
                
                evidence = Evidence(
                    code=code,
                    code_type="HCPCS",
                    method_name=self.method_name,
                    method_version=self.method_version,
                    raw_output=reasoning,
                    normalized_confidence=confidence,
                    metadata={
                        "model_name": MODEL_NAME,
                        "model_version": MODEL_VERSION,
                        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                        "matching_keywords": list(matching),
                        "num_matches": num_matches
                    }
                )
                
                evidences.append(evidence)
        
        return evidences
