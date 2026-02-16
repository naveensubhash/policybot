import pandas as pd
import re
from typing import List, Set
from src.methods.base_method import InferenceMethod
from src.schemas import Evidence


class MockLLMMethod(InferenceMethod):
    """
    Mock LLM inference method using keyword matching.
    
    Optimized for real policy documents with administrative content.
    Simulates LLM behavior by matching medical keywords from HCPCS descriptions
    against policy text. In production, this would be replaced with actual
    LLM API calls for semantic understanding.
    """

    def __init__(self, hcpcs_df: pd.DataFrame):
        """
        Initialize the mock LLM method.
        
        Args:
            hcpcs_df: DataFrame containing HCPCS codes and descriptions
        """
        self.hcpcs_df = hcpcs_df
        self.method_name = "mock_llm_reasoning"
        self.method_version = "1.0"
        
        # Mock-specific model configuration
        self.model_name = "mock-gpt-4"
        self.model_version = "1.0"
        self.prompt_template_version = "1.0"
        
        # Comprehensive stop words list
        self.stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
            'some', 'such', 'than', 'too', 'very', 'just', 'only',
            # Document/administrative words
            'page', 'policy', 'number', 'reference', 'effective', 'date',
            'version', 'revision', 'document', 'section', 'appendix',
            'clinical', 'procedure', 'service', 'treatment', 'patient', 'medical',
            'copyright', 'rights', 'reserved', 'inc', 'llc', 'corp'
        }
        
        # Medical/clinical terms that are valuable
        self.medical_terms = {
            # Imaging
            'mri', 'magnetic', 'resonance', 'imaging', 'scan', 'ct', 'pet', 
            'ultrasound', 'xray', 'x-ray', 'mammogram', 'fluoroscopy',
            # Procedures
            'surgery', 'surgical', 'biopsy', 'injection', 'infusion', 'transfusion',
            'transplant', 'dialysis', 'catheter', 'endoscopy', 'laparoscopy',
            'anesthesia', 'intubation', 'ventilation',
            # Diagnostics
            'laboratory', 'lab', 'test', 'screening', 'diagnostic', 'assessment',
            'evaluation', 'examination', 'echocardiogram', 'electrocardiogram', 
            'ekg', 'ecg', 'eeg',
            # Treatments
            'chemotherapy', 'radiation', 'radiotherapy', 'immunotherapy',
            'physical', 'occupational', 'speech', 'rehabilitation',
            # Medications
            'drug', 'medication', 'vaccine', 'vaccination', 'immunization',
            'administration',
            # Conditions/body systems
            'cardiac', 'cardiovascular', 'pulmonary', 'respiratory', 'neurological',
            'orthopedic', 'oncology', 'radiology', 'pathology', 'dermatology',
            'gastroenterology', 'urology', 'nephrology', 'endocrine', 'diabetes',
            'hypertension', 'asthma', 'copd', 'pneumonia', 'influenza', 'covid',
            # Specific drugs/treatments
            'fremanezumab', 'ajovy', 'migraine', 'headache', 'botox', 'botulinum'
        }

    def _clean_text(self, text: str) -> str:
        """Clean policy text by removing obvious non-medical content"""
        # Remove dates (MM/DD/YYYY format)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{4}', '', text)
        # Remove reference numbers like "HIM.PA.SP66"
        text = re.sub(r'\b[A-Z]{2,}\.[A-Z]{2,}\.[A-Z0-9]+\b', '', text)
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        # Remove asterisks and special characters at start of words
        text = re.sub(r'[*#@]\w+', '', text)
        return text

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful medical keywords from text"""
        # Clean the text first
        text = self._clean_text(text)
        
        words = text.lower().split()
        keywords = set()
        
        for word in words:
            # Clean punctuation
            clean = word.strip('.,;:!?()[]{}"\'-/')
            
            # Keep if:
            # - 3+ characters
            # - Not a stop word
            # - Not pure digits
            # - Doesn't look like a code (e.g., "03a")
            if (len(clean) >= 3 and 
                clean not in self.stop_words and 
                not clean.isdigit() and
                not re.match(r'^\d+[a-z]$', clean)):
                keywords.add(clean)
        
        return keywords

    def infer(self, policy_text: str) -> List[Evidence]:
        """
        Infer HCPCS codes using keyword matching on real policy documents.
        
        Rules:
        - Need at least 2 matching keywords
        - At least 1 must be a recognized medical term
        - Confidence based on number and quality of matches
        """
        evidences = []
        
        if not policy_text or not policy_text.strip():
            return evidences
        
        # Extract keywords
        policy_keywords = self._extract_keywords(policy_text)
        
        if not policy_keywords:
            return evidences
        
        # Identify which policy keywords are medical terms
        policy_medical_terms = policy_keywords.intersection(self.medical_terms)
        
        # Match against each HCPCS code
        for _, row in self.hcpcs_df.iterrows():
            code = str(row['code'])
            description = str(row['description'])
            
            # Extract description keywords
            desc_keywords = self._extract_keywords(description)
            
            # Find ALL matches
            all_matches = policy_keywords.intersection(desc_keywords)
            
            # Find medical term matches
            medical_matches = all_matches.intersection(self.medical_terms)
            
            num_total_matches = len(all_matches)
            num_medical_matches = len(medical_matches)
            
            # Require: at least 2 total matches AND at least 1 medical term
            if num_total_matches >= 2 and num_medical_matches >= 1:
                # Calculate confidence
                # Base: 0.5 + 0.1 per match
                # Bonus: 0.15 per medical term match
                base_confidence = 0.5 + (num_total_matches * 0.1)
                medical_bonus = num_medical_matches * 0.15
                confidence = min(base_confidence + medical_bonus, 0.95)
                
                # Format matched terms (prioritize medical terms)
                if medical_matches:
                    matched_terms = ', '.join(sorted(list(medical_matches))[:3])
                    if len(all_matches) > len(medical_matches):
                        other_matches = all_matches - medical_matches
                        matched_terms += ' + ' + ', '.join(sorted(list(other_matches))[:2])
                else:
                    matched_terms = ', '.join(sorted(list(all_matches))[:5])
                
                reasoning = (
                    f"Found {num_total_matches} matching keywords "
                    f"({num_medical_matches} medical terms: {matched_terms}) "
                    f"between policy and HCPCS {code}: '{description}'"
                )
                
                evidence = Evidence(
                    code=code,
                    code_type="HCPCS",
                    method_name=self.method_name,
                    method_version=self.method_version,
                    raw_output=reasoning,
                    normalized_confidence=confidence,
                    metadata={
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "prompt_template_version": self.prompt_template_version,
                        "matching_keywords": list(all_matches)[:10],
                        "medical_term_matches": list(medical_matches),
                        "num_matches": num_total_matches,
                        "num_medical_matches": num_medical_matches
                    }
                )
                
                evidences.append(evidence)
        
        return evidences