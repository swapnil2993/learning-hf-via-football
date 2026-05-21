import os
from transformers import pipeline
# Import your common book utilities from Assignment 1
from read_book import extract_book_text, chunk_text

def run_tactical_search_engine():
    # 1. Load and chunk the complete book text
    raw_book_text = extract_book_text()
    book_chunks = chunk_text(raw_book_text, chunk_size=600, overlap=100)
    print(f"✅ Extracted {len(book_chunks)} overlapping chunks from the book.\n")
    
    # 2. Use the universally supported text-generation pipeline
    print("🤖 Initializing Local Tactical AI Reader (SmolLM2)...")
    generator = pipeline(
        "text-generation", 
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        device_map="auto"
    )
    print("⚡ System Ready! Type your question below.\n")
    
    # 3. Interactive Terminal Loop
    while True:
        user_query = input("⚽ Ask a tactical question (or type 'exit' to quit): ")
        if user_query.strip().lower() == 'exit':
            print("Shutting down the oracle. Goodbye!")
            break
            
        if not user_query.strip():
            continue
            
        print("🧠 Scanning the chapters for tactical evidence...")
        
        # 4. Keyword Fast-Pass Filter to select the most relevant chunk
        best_chunk = ""
        query_words = [word.lower() for word in user_query.split() if len(word) > 4]
        
        for chunk in book_chunks:
            if query_words and any(kw in chunk.lower() for kw in query_words):
                best_chunk = chunk
                break  # Grab the first matching evidence window
        
        if not best_chunk:
            # Fallback to the first descriptive chapter chunk if keyword search yields nothing
            best_chunk = book_chunks[min(5, len(book_chunks)-1)]
            
        # 5. Format a clean chat template instruction for the generative model
        messages = [
            {
                "role": "system",
                "role": "user",
                "content": f"Context: {best_chunk}\n\nQuestion: {user_query}\n\nAnswer the question concisely using only the text context provided above."
            }
        ]
        
        # 6. Execute model text generation
        output = generator(messages, max_new_tokens=60, temperature=0.1, do_sample=False)
        ai_response = output[0]['generated_text'][-1]['content'].strip()
        
        print(f"\n🎯 Answer: {ai_response}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
  run_tactical_search_engine()
