import os
import re
from collections import Counter
import fitz  # PyMuPDF
from docx import Document
import pandas as pd

# A basic set of English stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    "did", "doing", "i\'d", "i'll", "i'm", "i've", "he'd", "he'll", "he's", "her'd",
    "her's", "his'd", "his'll", "his'm", "his're", "his've", "it'd", "it'll", "it's",
    "it've", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're",
    "we've", "you'd", "you'll", "you're", "you've", "i'd", "i'll", "i'm", "i've",
    "can'd", "couldn't", "shouldn't", "wouldn't"
}

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text

def clean_text(text):
    # Remove non-alphabetic characters (except spaces and hyphens)
    # We'll keep hyphens to handle words like 'well-being'
    text = re.sub(r'[^a-zA-Z\s\-]', ' ', text)
    # Replace multiple spaces/hyphens with a single space
    text = re.sub(r'[\s\-]+', ' ', text)
    return text.lower().strip()

def get_word_frequencies(base_dir):
    word_counts = Counter()
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file_path = os.path.join(root, file)
            text = ""
            if file.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)
            
            if text:
                cleaned = clean_text(text)
                words = cleaned.split()
                # Filter stopwords and very short words
                filtered_words = [w for w in words if w not in STOPWORDS and len(w) > 1]
                word_counts.update(filtered_words)
                
    return word_counts

def main(data_dir, output_file):
    print(f"Scanning directory: {data_dir}")
    counts = get_word_frequencies(data_dir)
    
    if not counts:
        print("No words found. Check your directory or file formats.")
        return

    df = pd.DataFrame(counts.items(), columns=['word', 'frequency'])
    # Sort by frequency descending
    df = df.sort_values(by='frequency', ascending=False).reset_index(drop=True)
    
    df.to_csv(output_file, index=False)
    print(f"Success! Found {len(df)} unique words. Saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract word frequencies from PDF and DOCX files.")
    parser.add_argument("data_dir", help="Directory to scan")
    parser.add_argument("output_file", help="Output CSV file path")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_file)
