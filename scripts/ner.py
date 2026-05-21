import os
import pandas as pd
from transformers import pipeline

from read_book import extract_book_text, chunk_text

def main():    
    try:
        # Re-use our modular utilities
        raw_text = extract_book_text()
        # NER models typically have smaller context limits (use chunk_size=300)
        book_chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
        print(f"✅ Extracted {len(book_chunks)} processing chunks for Named Entity Recognition.")
    except FileNotFoundError as e:
        print(f"⚠️ Error: {e}")
        return

    print("🤖 Initializing Token Classification (NER) Pipeline...")
    # CRITICAL PARAMETER: aggregation_strategy="simple" groups sub-word tokens back into whole words
    ner_pipeline = pipeline(
        "ner", 
        model="dbmdz/bert-large-cased-finetuned-conll03-english", 
        aggregation_strategy="simple"
    )
    print("✅ Model loaded successfully!\n")

    extracted_entities = []

    # To save execution time, we will scan the first 30 chunks of the book as a laboratory test
    # You can expand this range to scan the entire book
    max_scan_chunks = min(30, len(book_chunks))
    print(f"⚽ Extracting legendary pioneers and clubs from the first {max_scan_chunks} book chunks...\n")

    for idx in range(max_scan_chunks):
        chunk = book_chunks[idx]
        
        # Run token classification
        pipeline_output = ner_pipeline(chunk)
        
        for entity in pipeline_output:
            # Filter to keep only People (PER) and Organizations/Clubs (ORG)
            if entity['entity_group'] in ["PER", "ORG"]:
                extracted_entities.append({
                    "Chunk_Index": idx,
                    "Entity_Name": entity['word'].strip(),
                    "Type": "👤 Person/Manager" if entity['entity_group'] == "PER" else "🏟️ Club/Country",
                    "Confidence": f"{entity['score']:.2%}"
                })

    # Clean the dataset: drop duplicates to build a precise tactical roster
    df = pd.DataFrame(extracted_entities)
    if not df.empty:
        df = df.drop_duplicates(subset=["Entity_Name", "Type"])
        
        # Save structured spreadsheet output
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/tactical_pioneers.csv", index=False)
        
        print("\n=== Sample of Extracted Roster ===")
        print(df.head(15).to_string(index=False))
        print(f"\n✅ Success! Extracted entities exported to 'data/tactical_pioneers.csv'")
    else:
        print("❌ No matching entities were found in the scanned chunks.")

if __name__ == "__main__":
    main()