import os
import pandas as pd
from transformers import pipeline

# Import your common book utilities from Assignment 1
from read_book import extract_book_text, chunk_text

def run_chapter_summarizer():
    
    try:
        # Load and slice the book text into larger summary context blocks
        raw_book_text = extract_book_text()
        book_chunks = chunk_text(raw_book_text, chunk_size=800, overlap=100)
        print(f"✅ Extracted {len(book_chunks)} text blocks for summarization.")
    except FileNotFoundError as e:
        print(f"⚠️ Error: {e}")
        return

    # 1. INITIALIZE PIPELINE WITHOUT HARDWARE FLAGS TO AVOID ERROR WARNINGS
    print("🤖 Initializing Local Tactical AI Summarizer (SmolLM2)...")
    summarizer = pipeline(
        "text-generation", 
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        clean_up_tokenization_spaces=False  # Bypasses the BPE tokenizer warning log
    )
    print("⚡ System Ready!\n")

    # Let's test a slice of chapters (chunks 15 to 17)
    start_chunk = min(15, len(book_chunks)-1)
    end_chunk = min(18, len(book_chunks))
    
    summary_reports = []

    print(f"⚽ Summarizing book chapters from chunk {start_chunk} to {end_chunk-1}...\n")
    for idx in range(start_chunk, end_chunk):
        context = book_chunks[idx]
        
        # 2. FORMAT A CHAT TEMPLATE INSTRUCTION
        messages = [
            {
                "role": "user",
                "content": (
                    f"Text: {context}\n\n"
                    f"Summarize the main football tactical details from the text above "
                    f"into exactly three concise bullet points. Do not write any introduction."
                )
            }
        ]
        
        # 3. APPLY DYNAMIC GENERATION CONTROL PARAMETERS
        output = summarizer(
            messages,
            max_new_tokens=120,     
            temperature=0.1,        
            do_sample=False
        )
        
        # 4. SAFE ROBUST OUTPUT PARSING
        raw_output = output[0]['generated_text']
        
        # If the pipeline returns a conversational chat list structure
        if isinstance(raw_output, list):
            summary_text = raw_output[-1]['content'].strip()
        # If it returns a single concatenated string back
        else:
            summary_text = str(raw_output).strip()
            # Clean up template echo if the system repeats the prompt context
            if "into exactly three concise bullet points." in summary_text:
                summary_text = summary_text.split("into exactly three concise bullet points.")[-1].strip()
        
        print(f"📝 --- Chapter Chunk {idx} Tactical Summary ---")
        print(f"{summary_text}\n")
        print("-" * 60)
        
        summary_reports.append({
            "Chunk_Index": idx,
            "Tactical_Summary": summary_text
        })

    # Save outputs to data directory
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(summary_reports)
    df.to_csv("data/chapter_summaries.csv", index=False)
    print("📊 Summary metrics spreadsheet exported to 'data/chapter_summaries.csv'")

if __name__ == "__main__":
    run_chapter_summarizer()
