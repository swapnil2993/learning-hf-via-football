import os
import pandas as pd
from transformers import pipeline

from read_book import extract_book_text, chunk_text

def main():
    try:
        # Use the imported common utility functions
        raw_text = extract_book_text()
        book_chunks = chunk_text(raw_text, chunk_size=400, overlap=100)
        print(f"✅ Extracted {len(book_chunks)} processing chunks from the full book text.")
    except FileNotFoundError as e:
        print(f"⚠️ Error: {e}")
        return

    print("🤖 Initializing Zero-Shot Tactical Classifier...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    tactical_eras = [
      "Victorian 2-3-5 Pyramid", 
      "Chapman's W-M Formation", 
      "Catenaccio Defensive System",
      "Fluid Total Football", 
      "Modern High-Press Possession"
    ]

    timeline_data = []

    print("⚽ Running historical analysis mapping across the full text...\n")
    for idx, chunk in enumerate(book_chunks):
        output = classifier(chunk, candidate_labels=tactical_eras)
        
        # Access the first element of list outputs safely
        top_label = output['labels'][0]
        top_score = output['scores'][0]

        if top_score > 0.65:
            print(f"📍 Anchor Found in Chunk {idx} -> {top_label} ({top_score:.2%})")
            
            snippet = " ".join(chunk.split()[:15]) + "..."
            timeline_data.append({
                "Chunk_Index": idx,
                "Detected_Era": top_label,
                "Confidence": f"{top_score:.2%}",
                "Context_Snippet": snippet
            })

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(timeline_data)
    df.to_csv("data/tactical_timeline.csv", index=False)
    print(f"\n✅ Finished! Data successfully exported to 'data/tactical_timeline.csv'")

if __name__ == "__main__":
    main()
