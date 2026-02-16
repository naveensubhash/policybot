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
│  │ Method 2: RAG (TBD)  │   │
│  │ Method 3: ... (TBD)  │   │
│  └──────────────────────┘   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────┐
│  Aggregation    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Inferred Codes + Audit      │
└─────────────────────────────┘
```

### Design Principles

1. **Pluggable Methods**: Abstract base class pattern allows multiple inference methods
2. **Versioned Components**: All components track their version for reproducibility
3. **Audit Trail**: Complete provenance metadata (input hash, timestamps, method versions)
4. **Stable Schema**: Output schema supports evolution without breaking consumers
5. **Uncertainty Handling**: Confidence scores at evidence and aggregated levels

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
    "inferred_codes": [
      {
        "code": "G0008",
        "aggregated_confidence": 0.85,
        "selected": true,
        "justification": "Found 3 matching terms (influenza, vaccine, administration)..."
      }
    ],
    "evidence": [
      {
        "code": "G0008",
        "code_type": "HCPCS",
        "method_name": "mock_llm_reasoning",
        "method_version": "1.0",
        "raw_output": "...",
        "normalized_confidence": 0.85,
        "metadata": {...}
      }
    ],
    "audit": {
      "pipeline_version": "1.0",
      "run_id": "abc-123-def-456",
      "input_hash": "sha256...",
      "timestamp": "2026-02-15T10:30:00.000000",
      "methods_used": [...],
      "num_codes_inferred": 1,
      "num_codes_selected": 1
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

### Execute
```bash
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json
```

### Example
```bash
# Process sample policies
python run_pipeline.py -input policy_snippets.csv -output inferred_codes.json

# Output:
# Loading policies from policy_snippets.csv...
# Loading HCPCS reference data...
# Initializing inference engine...
# Processing 100 policies...
# ==================================================
# Pipeline Complete!
# ==================================================
# Policies processed: 100
# Total codes inferred: 247
# Codes meeting threshold: 189
# Output saved to: inferred_codes.json
# ==================================================
```

## Current Implementation: Mock LLM Method

The current `MockLLMMethod` uses keyword matching to simulate LLM behavior:

1. Extracts keywords from policy text (removing stop words)
2. Compares against HCPCS code descriptions
3. Identifies codes with ≥2 matching keywords
4. Assigns confidence scores based on number of matches

**This is intentionally simple** to focus on system design. In production, this would be replaced with:
- Real LLM API calls (GPT-4, Claude, etc.)
- RAG (Retrieval Augmented Generation) with policy database
- Embedding-based similarity search
- Ensemble methods combining multiple approaches

## Evolution: Supporting Multiple Methods

The architecture already supports multiple inference methods. To add a new method:

### 1. Implement the InferenceMethod Interface
```python
from src.methods.base_method import InferenceMethod
from src.schemas import Evidence

class RAGMethod(InferenceMethod):
    def __init__(self, vector_store):
        self.vector_store = vector_store
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
rag_method = RAGMethod(vector_store)

engine = InferenceEngine(methods=[mock_method, rag_method])
```

### 3. Output Schema Stays the Same!

The output already tracks which method produced each piece of evidence:
```json
{
  "evidence": [
    {
      "code": "70551",
      "method_name": "mock_llm_reasoning",  // From Mock
      "method_version": "1.0",
      "normalized_confidence": 0.85
    },
    {
      "code": "70551",
      "method_name": "rag_retrieval",  // From RAG
      "method_version": "1.0",
      "normalized_confidence": 0.92
    }
  ]
}
```

The aggregation layer combines evidence from all methods (taking max confidence).

**V1 consumers continue to work without any changes.**

## Configuration

Configurable parameters in `src/config.py`:
```python
CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence to select a code
PIPELINE_VERSION = "1.0"
AGGREGATION_VERSION = "max_confidence_v1"
REFERENCE_HCPCS_VERSION = "2026.01"
```

## Future Enhancements

- **Real LLM Integration**: Replace mock with GPT-4/Claude API calls
- **RAG System**: Add vector database for policy retrieval
- **Embedding Similarity**: Use sentence embeddings for semantic matching
- **Ensemble Methods**: Weighted voting across multiple methods
- **Confidence Calibration**: Train calibration models for better uncertainty estimates
- **API Wrapper**: REST API for real-time inference
- **Batch Processing**: Optimize for large-scale processing
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
│   │   └── mock_llm_method.py # Mock implementation using keyword matching
│   └── pipeline/
│       ├── inference_engine.py # Orchestrates inference methods
│       └── aggregator.py       # Combines evidence into final codes
├── run_pipeline.py            # CLI entrypoint
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Design Decisions

### 1. Text-Only Input
The service accepts only policy text as input. No external identifiers (policy_id) are required. This makes the service:
- Stateless and reusable
- Independent of policy tracking systems
- Suitable for real-time API deployment

IDs are generated internally (via content hash or UUID) purely for audit purposes.

### 2. Evidence-Based Architecture
Rather than having methods directly output "final answers," each method produces `Evidence` objects. This allows:
- Transparency: See reasoning from each method
- Flexibility: Different aggregation strategies
- Debuggability: Trace why a code was selected

### 3. Confidence Thresholding
Codes are marked as `selected` based on a confidence threshold (default: 0.65). This allows downstream consumers to:
- Use high-confidence codes immediately
- Review lower-confidence codes manually
- Adjust threshold based on their risk tolerance

### 4. Comprehensive Audit Trail
Every result includes:
- `run_id`: Unique identifier for this inference run
- `input_hash`: SHA-256 hash of input text for reproducibility
- Version information: Pipeline, methods, data sources, aggregation strategy
- Timestamp: When inference was performed

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

For questions or issues, contact: hello@policybot.app