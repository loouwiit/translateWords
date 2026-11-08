import subprocess
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Vocab Extractor & Refiner Orchestrator")
    parser.add_argument("--data_dir", required=True, help="Directory containing your study materials")
    parser.add_argument("--model", default="llama3", help="Ollama model to use for translation")
    parser.add_argument("--output_dir", default="output", help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for Ollama requests")

    args = parser.parse_args()

    # Create project paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    raw_csv = os.path.join(output_dir, "raw_vocab.csv")
    refined_csv = os.path.join(output_dir, "refined_vocab.csv")

    # 1. Extraction Phase
    print("\n=== Phase 1: Extraction ===")
    if os.path.exists(raw_csv) and os.path.getsize(raw_csv) > 0:
        print(f"Found existing raw vocabulary: {raw_csv}. Skipping extraction to save time.")
    else:
        extractor_script = os.path.join(project_dir, "extractor.py")
        try:
            subprocess.run([sys.executable, extractor_script, data_dir, raw_csv], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Extraction failed: {e}")
            sys.exit(1)

    if not os.path.exists(raw_csv) or os.path.getsize(raw_csv) == 0:
        print("No raw vocabulary was generated. Exiting.")
        sys.exit(1)

    # 2. Refinement Phase
    print("\n=== Phase 2: Refinement (using Ollama) ===")
    refiner_script = os.path.join(project_dir, "refiner.py")
    try:
        subprocess.run([
            sys.executable, 
            refiner_script, 
            raw_csv, 
            refined_csv, 
            "--model", args.model, 
            "--batch_size", str(args.batch_size)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Refinement failed: {e}")
        sys.exit(1)

    print("\n=== All tasks completed! ===")
    print(f"Raw data: {raw_csv}")
    print(f"Final word list: {refined_csv}")

if __name__ == "__main__":
    main()
