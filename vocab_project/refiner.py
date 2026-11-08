import pandas as pd
import requests
import json
import time
import argparse
import sys
import os

def call_ollama(model, prompt):
    """Calls the local Ollama API."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful linguistic assistant. You output ONLY valid CSV data. No preamble, no explanation."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def refine_vocab(input_csv, output_csv, model, batch_size=10):
    progress_csv = output_csv.replace('refined_', 'progress_')
    
    # 1. Load original data
    df = pd.read_csv(input_csv)
    if df.empty:
        print("Input CSV is empty.")
        return

    # 2. Load progress if exists
    already_done = {}
    if os.path.exists(progress_csv):
        print(f"Resuming from existing progress: {progress_csv}")
        progress_df = pd.read_csv(progress_csv)
        # Ensure columns are correct
        if not progress_df.empty and 'word' in progress_df.columns and 'meaning' in progress_df.columns:
            already_done = dict(zip(progress_df['word'], progress_df['meaning']))
    else:
        print(f"No progress file found. Starting from scratch. New progress will be saved to: {progress_csv}")

    # 3. Filter out already done words
    # We work on a copy of the dataframe that only contains words not in already_done
    remaining_df = df[~df['word'].isin(already_done.keys())].copy()
    
    total_remaining = len(remaining_df)
    if total_remaining == 0:
        print("All words have already been processed. Generating final output...")
        # If everything is already done, just merge progress and original
        final_df = pd.merge(df, pd.DataFrame(list(already_done.items()), columns=['word', 'meaning']), on='word', how='left')
        final_df = final_df[['word', 'meaning', 'frequency']].sort_values(by='frequency', ascending=False)
        final_df.to_csv(output_csv, index=False)
        print(f"Final output generated: {output_csv}")
        return

    print(f"Words remaining to process: {total_remaining}")

    # 4. Processing loop
    for i in range(0, total_remaining, batch_size):
        batch = remaining_df.iloc[i:i+batch_size]
        words_list = batch['word'].tolist()
        
        # Construct the prompt
        words_str = "\n".join(words_list)
        prompt = f"""Please translate the following English words into Chinese. 
For each word, provide its core meanings suitable for reading comprehension. 
Merge similar meanings. 
Output ONLY in the following CSV format:
word,meaning

Words to translate:
{words_str}"""

        print(f"Processing batch {i//batch_size + 1}/{(total_remaining + batch_size - 1)//batch_size} ({i}/{total_remaining})...")
        
        result = call_ollama(model, prompt)
        
        new_batch_results = []
        if result:
            lines = result.strip().split('\n')
            for line in lines:
                # Skip header or empty lines
                if not line.strip() or line.lower().startswith('word'):
                    continue
                
                if ',' in line:
                    parts = line.split(',', 1)
                    word = parts[0].strip()
                    meaning = parts[1].strip()
                    
                    # Only add if it's actually in our current batch to avoid hallucinations
                    if word in words_list:
                        new_batch_results.append({'word': word, 'meaning': meaning})
        
        # 5. Append new results to progress file immediately
        if new_batch_results:
            new_batch_df = pd.DataFrame(new_batch_results)
            # Append to CSV: header only if file doesn't exist
            file_exists = os.path.exists(progress_csv)
            new_batch_df.to_csv(progress_csv, mode='a', index=False, header=not file_exists)
        else:
            print(f"Warning: Batch {i//batch_size + 1} yielded no valid results.")

        # Small delay to be nice to the system
        time.sleep(0.5)

    # 6. Finalization: After loop (or if user interrupts and restarts), 
    # the user should run again to finish, OR if we finished all, merge now.
    # To make it seamless, we only generate the final 'refined_' file if we think we're done.
    # However, for a long-running task, it's safer to just let the user run it again 
    # until 'remaining_df' is empty.
    
    print("\nBatch completed. If there are still words remaining, run the command again to resume.")
    print(f"Current progress saved in: {progress_csv}")
    print("To generate the final file, run again when all words are processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refine vocabulary using Ollama.")
    parser.add_argument("input_csv", help="Path to raw_vocab.csv")
    parser.add_argument("output_csv", help="Path to refined_vocab.csv")
    parser.add_argument("--model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for Ollama requests (default: 5)")
    
    args = parser.parse_args()
    refine_vocab(args.input_csv, args.output_csv, args.model, args.batch_size)
