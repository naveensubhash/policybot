from abc import ABC, abstractmethod
from typing import List
from src.schemas import Evidence


class InferenceMethod(ABC):
    """
    Abstract base class for HCPCS code inference methods.
    
    All inference methods must implement the infer() method which takes
    policy text as input and returns a list of Evidence objects.
    """

    @abstractmethod
    def infer(self, policy_text: str) -> List[Evidence]:
        """
        Infer HCPCS codes from policy text.
        
        Args:
            policy_text: The policy document text to analyze
            
        Returns:
            List of Evidence objects containing inferred codes and metadata
        """
        pass