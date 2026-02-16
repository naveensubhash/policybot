import argparse
import json
from src.data.hcpcs_loader import load_hcpcs
from src.data.policy_loader import load_policies
from src.methods.mock_llm_method import MockLLMMethod
from src.pipeline.inference_engine import InferenceEngine


def serialize_result(result):
    """Convert dataclass objects to dictionaries for JSON serialization"""
    return {
        "inferred_codes": [
            {
                "code": code.code,
                "aggregated_confidence": code.aggregated_confidence,
                "selected": code.selected,
                "justification": code.justification
            }
            for code in result.inferred_codes
        ],
        "evidence": [
            {
                "code": ev.code,
                "code_type": ev.code_type,
                "method_name": ev.method_name,
                "method_version": ev.method_version,
                "raw_output": ev.raw_output,
                "normalized_confidence": ev.normalized_confidence,
                "metadata": ev.metadata
            }
            for ev in result.evidence
        ],
        "audit": result.audit
    }


def main(input_path: str, output_path: str):
    """
    Main pipeline entrypoint.
    
    Loads policy data, runs inference, and saves results.
    
    Args:
        input_path: Path to input CSV with 'policy_text' column
        output_path: Path to save JSON output
    """
    print(f"Loading policies from {input_path}...")
    policies_df = load_policies(input_path)
    
    print(f"Loading HCPCS reference data...")
    hcpcs_df = load_hcpcs("hcpcs.csv")
    
    print(f"Initializing inference engine...")
    method = MockLLMMethod(hcpcs_df)
    engine = InferenceEngine(methods=[method])
    
    print(f"Processing {len(policies_df)} policies...")
    results = []
    
    for idx, row in policies_df.iterrows():
        policy_text = row["policy_text"]
        
        # Run inference
        result = engine.run(policy_text)
        
        # Serialize result
        result_dict = serialize_result(result)
        
        # Add row index for tracking
        result_dict["row_index"] = int(idx)
        
        results.append(result_dict)
        
        # Progress update
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(policies_df)} policies...")
    
    print(f"Saving results to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary statistics
    total_codes = sum(len(r["inferred_codes"]) for r in results)
    selected_codes = sum(
        sum(1 for code in r["inferred_codes"] if code["selected"]) 
        for r in results
    )
    
    print("\n" + "="*50)
    print("Pipeline Complete!")
    print("="*50)
    print(f"Policies processed: {len(results)}")
    print(f"Total codes inferred: {total_codes}")
    print(f"Codes meeting threshold: {selected_codes}")
    print(f"Output saved to: {output_path}")
    print("="*50)


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