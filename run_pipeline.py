import argparse
import json
import os
from src.data.hcpcs_loader import load_hcpcs
from src.data.policy_loader import load_policies
from src.methods.mock_llm_method import MockLLMMethod
from src.methods.direct_match_groq_method import DirectMatchGroqMethod
from src.methods.rag_groq_method import RAGGroqMethod
from src.pipeline.inference_engine import InferenceEngine


def serialize_result(result, row_idx):
    """Convert PolicyInferenceResult to clean JSON format"""
    return {
        "row_index": row_idx,
        "codes": [
            {
                "code": code.code,
                "confidence": round(code.aggregated_confidence, 3),
                "justification": code.justification,
                "provenance": {
                    **code.provenance,
                    "runtime": {
                        "inference_timestamp": result.metadata["timestamp"],
                        "pipeline_version": result.metadata["pipeline_version"],
                        "run_id": result.metadata["run_id"]
                    },
                    "input": {
                        "policy_text_hash": result.metadata["input_hash"]
                    }
                }
            }
            for code in result.inferred_codes
        ],
        "summary": {
            "total_codes_found": result.metadata["num_codes_above_threshold"],
            "confidence_threshold": result.metadata["confidence_threshold"],
            "methods_used": [m["method_name"] for m in result.metadata["methods_used"]]
        }
    }


def main(input_path: str, output_path: str):
    """
    Main pipeline entrypoint.
    
    Args:
        input_path: Path to input CSV with 'policy_text' column
        output_path: Path to save JSON output
    """
    print(f"Loading policies from {input_path}...")
    policies_df = load_policies(input_path)
    
    print(f"Loading HCPCS reference data...")
    hcpcs_df = load_hcpcs("hcpcs.csv")
    
    print(f"\nInitializing inference methods...")
    
    # ========================================
    # 🎯 ADD/REMOVE METHODS HERE
    # ========================================
    methods = []
    
    # Method 1: Mock (keyword matching)
    print("  → Adding Mock LLM method (keyword matching)")
    methods.append(MockLLMMethod(hcpcs_df))
    
    # Method 2: Direct Match + Groq (similarity to HCPCS descriptions)
    if os.environ.get("GROQ_API_KEY"):
        print("  → Adding Direct-Match Groq method (HCPCS similarity + LLM)")
        methods.append(DirectMatchGroqMethod(hcpcs_df, api_key=os.environ.get("GROQ_API_KEY")))
    else:
        print("  ⚠️  Skipping Groq method (GROQ_API_KEY not set)")
    

    
    # Method 3: Add more methods here in the future!
    # methods.append(EmbeddingMethod(hcpcs_df))
    # methods.append(AnotherMethod(hcpcs_df))
    
    # ========================================
    
    if not methods:
        print("❌ No inference methods available!")
        return
    
    print(f"\n✓ Using {len(methods)} inference method(s)")
    
    # Initialize engine with all methods
    engine = InferenceEngine(methods=methods, hcpcs_df=hcpcs_df)
    
    print(f"\nProcessing {len(policies_df)} policies...")
    results = []
    
    for idx, row in policies_df.iterrows():
        policy_text = row["policy_text"]
        
        print(f"\n{'='*60}")
        print(f"Policy {idx + 1}/{len(policies_df)}")
        print(f"{'='*60}")
        
        # Run inference (all methods automatically)
        result = engine.run(policy_text)
        
        # Serialize result
        result_dict = serialize_result(result, int(idx))
        
        results.append(result_dict)
        
        # Show summary
        num_codes = len(result.inferred_codes)
        methods_used = [m["method_name"] for m in result.metadata["methods_used"]]
        print(f"\n✓ Found {num_codes} codes above threshold")
        print(f"  Methods: {', '.join(methods_used)}")
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary statistics
    total_codes = sum(r["summary"]["total_codes_found"] for r in results)
    all_methods = set()
    for r in results:
        all_methods.update(r["summary"]["methods_used"])
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Policies processed: {len(results)}")
    print(f"Total codes above threshold: {total_codes}")
    print(f"Methods used: {', '.join(all_methods)}")
    print(f"Output saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HCPCS Inference Pipeline - Infer medical procedure codes from policy text"
    )
    parser.add_argument(
        "-input", 
        required=True,
        help="Path to input CSV file with 'policy_text' column"
    )
    parser.add_argument(
        "-output", 
        required=True,
        help="Path to output JSON file"
    )
    
    args = parser.parse_args()
    
    main(args.input, args.output)