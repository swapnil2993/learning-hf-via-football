import os
from transformers import pipeline

def run_tactical_simulator():
    # 1. INITIALIZE THE CORE TEXT GENERATION PIPELINE CLEANLY
    print("🤖 Initializing Local Tactical Simulation Engine (SmolLM2)...")
    simulator = pipeline(
        "text-generation", 
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        clean_up_tokenization_spaces=False
    )
    print("⚡ System Ready!\n")

    print("🎭 Welcome to the 'Inverted Pyramid' Alternative History Lab!")
    print("Type a 'What If' premise or choose from the examples below:\n")
    print("💡 Example: If Arrigo Sacchi coached a modern team, his pressing rules would change because...")

    # 2. INTERACTIVE LOOP
    while True:
        user_prompt = input("\n🔮 Enter your 'What If' tactical premise (or type 'exit'): ")
        if user_prompt.strip().lower() == 'exit':
            print("Shutting down the simulation vault. Goodbye!")
            break
            
        if not user_prompt.strip():
            continue
            
        print("🧠 Simulating alternative history timeline...")

        # 3. FORMAT THE CHAT TEMPLATE INSTRUCTION
        messages = [
            {
                "role": "user",
                "content": (
                    f"Premise: {user_prompt}\n\n"
                    f"Act as a professional football data analyst deeply inspired by Jonathan Wilson's writing style. "
                    f"Continue writing the premise into a fascinating, deep tactical paragraph. "
                    f"Focus on formation shapes, space, and historical logic. Keep it under 150 words."
                )
            }
        ]

        # 4. WRAP GENERATION KEYS INSIDE THE MODEL ARGS DICTIONARY TO FIX THE DEPRECATION
        gen_kwargs = {
            "max_new_tokens": 200,      
            "temperature": 0.75,        
            "do_sample": True,          
            "top_k": 50,                
            "top_p": 0.92,              
            "no_repeat_ngram_size": 3   
        }

        try:
            # Pass parameters explicitly using kwargs unpacking (**gen_kwargs)
            output = simulator(messages, **gen_kwargs)
            
            # 5. FIX THE TYPEERROR BY UNWRAPPING THE LIST FIRST
            # output is a list: [{'generated_text': [...]}]
            raw_output = output[0]['generated_text']
            
            # Extract content text from the conversational message payload safely
            if isinstance(raw_output, list):
                simulation_text = raw_output[-1]['content'].strip()
            else:
                simulation_text = str(raw_output).strip()

            print("\n📜 --- Generated Tactical Timeline Alternative ---")
            print(simulation_text)
            print("-" * 60)
            
        except Exception as e:
            print(f"\n⚠️ Generation pipeline error caught: {e}")
            print("-" * 60)

if __name__ == "__main__":
    run_tactical_simulator()
