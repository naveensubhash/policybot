# HCPCS Inference Service v1

## Overview

This project implements a modular, auditable HCPCS inference service that:
- Accepts policy text as input
- Infers relevant HCPCS codes using pluggable inference methods
- Returns confidence scores for each code
- Provides full audit trail and provenance metadata for reproducibility

The system is designed for evolution: new inference methods can be added without breaking existing consumers.

## Architecture

### Core Components
```
┌─────────────────┐
│  Policy Text    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│   Inference Engine          │
│  ┌──────────────────────┐   │
│  │ Method 1: Mock LLM   │   │
│  │ Method 2: Groq LLM   │   │
│  │ Method 3: ... (TBD)  │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Aggregation    │
│  (Max Conf)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Codes + Provenance          │
└─────────────────────────────┘
```

### Design Principles

1. **Pluggable Methods**: Abstract base class pattern allows multiple inference methods
2. **Versioned Components**: All components track their version for reproducibility
3. **Audit Trail**: Complete provenance metadata per code
4. **Stable Schema**: Output schema supports evolution without breaking consumers
5. **Uncertainty Handling**: Confidence scores and thresholding

## Input Format

CSV file with a single `policy_text` column:
```csv
policy_text
"Coverage is provided for administration of influenza virus vaccine..."
"Magnetic resonance imaging of the brain is covered when medically necessary..."
```

## Output Format

JSON array with one object per input policy:
```json
[
  {
    "row_index": 0,
    "codes": [
      {
        "code": "70551",
        "confidence": 0.87,
        "justification": "Policy discusses MRI brain imaging without contrast",
        "provenance": {
          "method": {
            "name": "groq_llm",
            "type": "llm",
            "version": "1.0"
          },
          "model": {
            "name": "llama-3.3-70b-versatile",
            "provider": "groq",
            "tokens_used": 1523
          },
          "reference_data": {
            "hcpcs_version": "2026.01",
            "hcpcs_description": "Magnetic resonance imaging brain without contrast"
          },
          "runtime": {
            "inference_timestamp": "2026-02-16T14:22:00Z",
            "pipeline_version": "1.0",
            "run_id": "5c7a91ae-8623..."
          },
          "input": {
            "policy_text_hash": "sha256:ddf04a913..."
          }
        }
      }
    ],
    "summary": {
      "total_codes_found": 1,
      "confidence_threshold": 0.65,
      "methods_used": ["groq_llm"]
    }
  }
]
```

## Running the Pipeline

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Files

- `policy_snippets.csv` - Input policies (with `policy_text` column)
- `hcpcs.csv` - HCPCS code reference data (with `code` and `description` columns)

### Optional Files (for few-shot learning)

- `policies_cleaned.csv` - Training policy data
- `policies_cleaned_labels.csv` - Ground truth labels

### Execute
```bash
# Mock method (keyword matching)
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json

# Groq LLM (requires GROQ_API_KEY)
export GROQ_API_KEY="your_key_here"
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json --use-groq

# Ensemble (both methods)
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json --ensemble
```

### Example
```bash
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json

# Output:
# Loading policies from policy_snippets.csv...
# Loading HCPCS reference data...
# Initializing inference engine...
#   → Using Mock LLM (keyword matching)
# Processing 2 policies...
# 
# Policy 1/2:
#   → Found 3 codes above threshold
# 
# Policy 2/2:
#   → Found 5 codes above threshold
# 
# ==================================================
# Pipeline Complete!
# ==================================================
# Policies processed: 2
# Total codes above threshold: 8
# Output saved to: inferred_codes.json
# ==================================================
```

## Current Implementation

### Mock LLM Method

Uses keyword matching with medical term recognition:

1. Extracts keywords from policy text (removes administrative content)
2. Identifies medical terms (MRI, vaccine, surgery, etc.)
3. Matches against HCPCS code descriptions
4. Requires: 2+ matches AND 1+ medical term
5. Assigns confidence based on match quality

**This is intentionally simple** to focus on system design. In production, this would be replaced with real LLM inference.

### Groq LLM Method

Uses Llama 3.3 70B via Groq for semantic understanding:

1. Automatically loads training examples if available (`policies_cleaned.csv`)
2. Uses few-shot learning when training data exists
3. Sends policy text + HCPCS codes to LLM
4. LLM returns relevant codes with reasoning
5. Validates codes against HCPCS reference

**Training data detection is automatic** - no flags needed!

## Evolution: Supporting Multiple Methods

The architecture already supports multiple inference methods. To add a new method:

### 1. Implement the InferenceMethod Interface
```python
from src.methods.base_method import InferenceMethod
from src.schemas import Evidence

class RAGMethod(InferenceMethod):
    def __init__(self, vector_store, hcpcs_df):
        self.vector_store = vector_store
        self.hcpcs_df = hcpcs_df
        self.method_name = "rag_retrieval"
        self.method_version = "1.0"
    
    def infer(self, policy_text: str) -> List[Evidence]:
        # Retrieve similar policies
        similar_policies = self.vector_store.search(policy_text, top_k=5)
        
        # Extract codes from similar policies
        # ...
        
        return evidences
```

### 2. Add to Inference Engine
```python
# In run_pipeline.py
mock_method = MockLLMMethod(hcpcs_df)
rag_method = RAGMethod(vector_store, hcpcs_df)

engine = InferenceEngine(methods=[mock_method, rag_method], hcpcs_df=hcpcs_df)
```

### 3. Output Schema Stays the Same!

The output already tracks which method produced each piece of evidence:
```json
{
  "provenance": {
    "method": {
      "name": "rag_retrieval",
      "type": "retrieval",
      "version": "1.0"
    }
  }
}
```

The aggregation layer combines evidence from all methods (taking max confidence).

**V1 consumers continue to work without any changes.**

## Configuration

Configurable parameters in `src/config.py`:
```python
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to include a code
PIPELINE_VERSION = "1.0"
AGGREGATION_VERSION = "max_confidence_v1"
REFERENCE_HCPCS_VERSION = "2026.01"
```

## Future Enhancements

- **RAG System**: Add vector database for policy retrieval
- **Embedding Similarity**: Use sentence embeddings for semantic matching
- **Ensemble Methods**: Weighted voting across multiple methods
- **Confidence Calibration**: Train calibration models for better uncertainty estimates
- **API Wrapper**: REST API for real-time inference
- **Evaluation Framework**: Precision/recall metrics against ground truth

## Project Structure
```
hcpcs_inference/
├── src/
│   ├── config.py              # Configuration constants
│   ├── schemas.py             # Data structures (Evidence, FinalCode, etc.)
│   ├── utils.py               # Utility functions (hashing, timestamps)
│   ├── data/
│   │   ├── hcpcs_loader.py    # Load HCPCS reference data
│   │   └── policy_loader.py   # Load policy data
│   ├── methods/
│   │   ├── base_method.py     # Abstract base class for inference methods
│   │   ├── mock_llm_method.py # Mock implementation using keyword matching
│   │   └── groq_llm_method.py # Real LLM via Groq API
│   └── pipeline/
│       ├── inference_engine.py # Orchestrates inference methods
│       └── aggregator.py       # Combines evidence into final codes
├── run_pipeline.py            # CLI entrypoint
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Design Decisions

### 1. Text-Only Input
The service accepts only policy text as input. No external identifiers required. This makes the service:
- Stateless and reusable
- Independent of policy tracking systems
- Suitable for real-time API deployment

### 2. Evidence-Based Architecture
Rather than having methods directly output "final answers," each method produces `Evidence` objects. This allows:
- Transparency: See reasoning from each method
- Flexibility: Different aggregation strategies
- Debuggability: Trace why a code was selected

### 3. Confidence Thresholding
Only codes above the threshold (default: 0.65) are included in the output. This allows:
- Clean output with only confident predictions
- Threshold can be adjusted in config
- Consumer can apply additional filtering using confidence scores

### 4. Comprehensive Provenance
Every code includes complete provenance:
- Which method found it
- What model was used (if LLM)
- When it was inferred
- Input hash for reproducibility
- HCPCS reference version

This enables:
- Reproducing historical results
- Debugging issues
- Compliance and regulatory requirements

## Evaluation

To evaluate against ground truth labels:
```python
import json
import pandas as pd

# Load results and labels
results = json.load(open('inferred_codes.json'))
labels_df = pd.read_csv('policies_cleaned_labels.csv')

# Compare inferred vs. ground truth codes
# Calculate precision, recall, F1 score
# ...
```

## License

Proprietary - Policybot Assessment

## Contact

For questions: hello@policybot.app