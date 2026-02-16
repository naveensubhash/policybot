import os
from typing import List, Tuple
import json
import pandas as pd
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.methods.base_method import InferenceMethod
from src.schemas import Evidence


class DirectMatchGroqMethod(InferenceMethod):
    """
    Direct HCPCS matching with LLM validation.
    
    Process:
    1. Calculate similarity between policy text and HCPCS descriptions
    2. Get top K most similar codes
    3. Send to LLM for validation and reasoning
    
    No training data needed - works directly with HCPCS reference.
    """

    def __init__(self, hcpcs_df, api_key: str = None, 
                 model: str = "llama-3.3-70b-versatile",
                 top_k_codes: int = 15,
                 similarity_threshold: float = 0.1):
        """
        Initialize the direct match Groq method.
        
        Args:
            hcpcs_df: DataFrame containing HCPCS codes and descriptions
            api_key: Groq API key
            model: Groq model to use
            top_k_codes: Number of top similar codes to retrieve
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.hcpcs_df = hcpcs_df
        self.method_name = "direct_match_groq"
        self.method_version = "1.0"
        self.model = model
        self.top_k_codes = top_k_codes
        self.similarity_threshold = similarity_threshold
        
        # Initialize Groq client
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY required")
        
        self.client = Groq(api_key=api_key)
        
        # Build TF-IDF index for HCPCS descriptions
        self._build_hcpcs_index()
    
    def _build_hcpcs_index(self):
        """
        Build TF-IDF index for HCPCS code descriptions.
        """
        print("  → Building HCPCS similarity index...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Vectorize all HCPCS descriptions
        descriptions = self.hcpcs_df['description'].fillna('').tolist()
        self.hcpcs_vectors = self.vectorizer.fit_transform(descriptions)
        
        print(f"  ✓ Indexed {len(descriptions)} HCPCS codes")
    
    def _find_similar_codes(self, policy_text: str) -> List[Tuple[str, str, float]]:
        """
        Find HCPCS codes with descriptions most similar to policy text.
        
        Args:
            policy_text: The policy document text
            
        Returns:
            List of tuples: (code, description, similarity_score)
        """
        try:
            # Vectorize policy text
            query_vector = self.vectorizer.transform([policy_text])
            
            # Calculate cosine similarity with all HCPCS descriptions
            similarities = cosine_similarity(query_vector, self.hcpcs_vectors)[0]
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[-self.top_k_codes:][::-1]
            
            similar_codes = []
            for idx in top_indices:
                similarity_score = similarities[idx]
                
                # Only include if above threshold
                if similarity_score < self.similarity_threshold:
                    continue
                
                code = self.hcpcs_df.iloc[idx]['code']
                description = self.hcpcs_df.iloc[idx]['description']
                
                similar_codes.append((code, description, float(similarity_score)))
            
            return similar_codes
            
        except Exception as e:
            print(f"  ⚠️  Error in similarity search: {e}")
            return []
    
    def _build_prompt(self, policy_text: str, similar_codes: List[Tuple[str, str, float]]) -> str:
        """
        Build prompt with policy text and most similar HCPCS codes.
        """
        # Build codes list
        codes_text = "Most relevant HCPCS codes (based on text similarity):\n\n"
        for code, description, similarity in similar_codes:
            codes_text += f"{code}: {description} (similarity: {similarity:.3f})\n"
        
        prompt = f"""You are a medical coding expert. Review the policy text and the relevant HCPCS codes below.

Policy Text:
{policy_text}

{codes_text}

Task: Determine which of the above codes are truly relevant to this policy. For each relevant code:
1. The code
2. Confidence (0.0 to 1.0) - how confident are you this code applies
3. Brief reasoning - explain WHY this code is relevant

Return ONLY a JSON array:
[
  {{"code": "XXXXX", "confidence": 0.85, "reasoning": "This code is relevant because..."}}
]

Important:
- Only include codes that are clearly relevant to the policy
- A high similarity score doesn't mean the code is relevant - use your medical knowledge
- If none of the codes are relevant, return: []
"""
        return prompt
    
    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from LLM response"""
        text = text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        start = text.find('[')
        end = text.rfind(']')
        
        if start != -1 and end != -1:
            return text[start:end+1]
        
        return text
    
    def infer(self, policy_text: str) -> List[Evidence]:
        """
        Infer HCPCS codes using direct similarity matching + LLM validation.
        
        Process:
        1. Find codes with similar descriptions to policy text
        2. Send to LLM for validation and reasoning
        """
        if not policy_text or not policy_text.strip():
            return []
        
        # Step 1: Find similar codes
        print(f"  → Searching for similar HCPCS codes...")
        similar_codes = self._find_similar_codes(policy_text)
        
        if not similar_codes:
            print(f"  ⚠️  No similar codes found (threshold={self.similarity_threshold})")
            return []
        
        print(f"  → Found {len(similar_codes)} similar codes:")
        for code, desc, sim in similar_codes[:5]:
            print(f"    - {code}: {desc[:50]}... (sim: {sim:.3f})")
        
        # Truncate policy text if needed
        max_policy_length = 3000
        if len(policy_text) > max_policy_length:
            policy_text = policy_text[:max_policy_length] + "\n...[truncated]"
        
        # Step 2: Build prompt with similar codes
        prompt = self._build_prompt(policy_text, similar_codes)
        
        # Step 3: Call LLM for validation
        try:
            print(f"  → Calling Groq API for validation of {len(similar_codes)} codes...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical coding expert. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=2000,
                top_p=0.9
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"  → Received response ({len(response_text)} chars)")
            
            # Parse JSON
            json_text = self._extract_json_from_response(response_text)
            codes_data = json.loads(json_text)
            
            if not isinstance(codes_data, list):
                print(f"  ⚠️  Warning: Expected JSON array")
                return []
            
            print(f"  → LLM validated {len(codes_data)} codes as relevant")
            
            # Convert to Evidence objects
            evidences = []
            
            # Create a lookup for similarity scores
            similarity_lookup = {code: sim for code, desc, sim in similar_codes}
            
            for item in codes_data:
                try:
                    code = str(item['code']).strip()
                    confidence = float(item['confidence'])
                    reasoning = str(item['reasoning'])
                    
                    # Validate code
                    if code not in self.hcpcs_df['code'].values:
                        print(f"  ⚠️  Skipping invalid code: {code}")
                        continue
                    
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # Get similarity score for this code
                    similarity_score = similarity_lookup.get(code, 0.0)
                    
                    evidence = Evidence(
                        code=code,
                        code_type="HCPCS",
                        method_name=self.method_name,
                        method_version=self.method_version,
                        raw_output=reasoning,
                        normalized_confidence=confidence,
                        metadata={
                            "model_name": self.model,
                            "api_provider": "groq",
                            "matching_method": "tfidf_description_similarity",
                            "description_similarity_score": similarity_score,
                            "num_candidate_codes": len(similar_codes),
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens
                        }
                    )
                    
                    evidences.append(evidence)
                    
                except (KeyError, ValueError, TypeError) as e:
                    print(f"  ⚠️  Error parsing: {e}")
                    continue
            
            print(f"  ✓ Created {len(evidences)} evidence objects")
            return evidences
            
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parse error: {e}")
            print(f"     Response: {response_text[:200]}...")
            return []
        
        except Exception as e:
            print(f"  ❌ API error: {e}")
            return []