# Inverted Pyramid NLP: Football Tactical Evolution Engine ⚽📐

A data-driven exploration of football's tactical history powered by Hugging Face transformers. This repository serves as an iterative learning laboratory to master modern NLP pipelines (`translation`, `zero-shot-classification`, `ner`, `question-answering`, `text-generation`) using structural insights, narratives, and concepts inspired by Jonathan Wilson's seminal book, *The Inverted Pyramid*.


## Assignment 1: The Tactical Time-Machine 🕰️⚽

### 📌 Problem Statement
**"How can we automatically map the chronological timeline of football's tactical evolution from a massive, unstructured book without manually reading it or training a custom AI model?"**

#### The Core Challenges:
1. **Unstructured Data Clutter:** Commercial book containers (`.epub`) are packed with raw HTML elements, formatting tags, and narrative prose that confuse standard keyword parsers.
2. **The Transformer Memory Wall:** Deep learning models cannot ingest an entire book at once due to strict context window limits (typically a few hundred tokens).
3. **Zero-Training Constraint:** Paragraphs must be categorized into high-level tactical philosophies instantly without spending time manually labeling data or fine-tuning models from scratch.

---

#### 🛠️ Technical Solution & Architecture
We solve this by utilizing a modular Python structure that combines local file parsing utilities with zero-shot transformer pipelines.

#### Workflow:
1. **Clean:** Ingest the raw `.epub` file, identify document sections, and strip out hidden layout scripts or XML/HTML tags using `BeautifulSoup`.
2. **Chunk:** Split the continuous string into sliding text windows of 400 words with a 100-word buffer overlap to preserve context across boundaries.
3. **Classify:** Feed every individual chunk into a `zero-shot-classification` pipeline running the `facebook/bart-large-mnli` model.
4. **Filter & Save:** Set an inference threshold (`confidence > 65%`) to ignore introductory filler text, and export the high-confidence historical anchors into a structured `.csv` datatable.

---


# Assignment 2: The Pioneer Extractor 👤🏟️

## 📌 Problem Statement
**"How can we automatically extract a structured roster of historical football figures (managers, players) and the entities (clubs, countries) they influenced directly from raw, unlabelled chapters?"**

### The Core Challenges:
1. **Sub-Word Tokenization Fragmentation:** Transformer models split unfamiliar, complex, or historic last names (e.g., *Lobanovskyi* or *Pozzo*) into strange fractional sub-tokens like `['Loba', '##novs', '##kyi']`. This breaks traditional entity filtering.
2. **Entity Distraction:** Text narratives are packed with generic nouns, numbers, and system terms. The engine must discard irrelevant text and isolate only the key tactical actors.
3. **Data Prototyping:** Extracting relationships sequentially via basic Python text matches fails to recognize contextual usage (e.g., distinguishing when a word is a person vs. a location).

---

## 🛠️ Technical Solution & Architecture
We solve this by passing the extracted text stream through a Token Classification architecture optimized for Named Entity Recognition (NER).

### Workflow:
1. **Modular Text Load:** Call the common utility `extract_book_text` from our shared `read_book.py` module to parse the continuous book string.
2. **Targeted Aggregation:** Initialize the `ner` pipeline using the high-capacity `dbmdz/bert-large-cased-finetuned-conll03-english` model weights.
3. **Reconstruct Whole Words:** Pass the explicit configuration `aggregation_strategy="simple"`. This instructs Hugging Face to intercept sub-tokens, combine them based on model probability maps, and return whole names cleanly.
4. **Filter & Deduplicate:** Parse the raw pipeline dictionary tokens, select entries matching `PER` (Person) and `ORG` (Organization), clear row duplicates, and output a clean spreadsheet roster.

---

# Assignment 3: The Inverted Pyramid Oracle 🔮⚽

## 📌 Problem Statement
**"How can we build an interactive search engine that allows users to ask open-ended questions about football history across a massive book, without crashing memory or losing sentence context?"**

### The Core Challenges:
1. **The Context Window Limitation:** Transformer models like RoBERTa cannot read a whole book at once. They have a strict limit (usually 512 tokens). If you feed them too much text, they throw an error or truncate the data.
2. **Broken Answers at Borders:** If you slice a book blindly into separate pages, a critical historical fact might get cut directly in half (e.g., the question is in Chunk 1 but the answer lands at the start of Chunk 2).
3. **Speed Hurdles on CPU:** Running deep learning inference on hundreds of paragraphs sequentially takes too long on standard laptops. The search must bypass irrelevant text paths instantly.

---

## 🛠️ Technical Solution & Architecture
We solve this by designing a **Sliding Window Chunking Engine** combined with a Reading Comprehension Question-Answering pipeline.

### Workflow:
1. **Global Text Extraction:** Use our shared `read_book.py` module to extract clean text from the entire `.epub` file.
2. **Sliding Window Slicing:** Break the text into chunks of 800 words, but add a 150-word **overlap buffer**. This ensures sentences that span borders are preserved whole in at least one chunk.
3. **Keyword Fast-Pass Filter:** Before giving a chunk to the AI, run a quick check. If the chunk does not contain words from the user's question, skip it. This speeds up CPU performance by up to 80%.
4. **Confidence Scanning:** Pass the active chunks to the `deepset/roberta-base-squad2` model. Loop over the text, find the answer snippet with the highest probability score, and print it to the user.

---
# Assignment 4: The Chapter Summarizer 📝📐

## 📌 Problem Statement
**"How can we automatically compress dense, narrative-heavy book chapters into concise tactical bullet points without losing historical precision or crashing local hardware resources?"**

### The Core Challenges:
1. **The Task Registry Phaseout:** Legacy task keys like `"summarization"` have been officially deprecated and removed from modern Hugging Face versions, making old code architectures obsolete.
2. **Deterministic Output Locking:** Standard language models are highly creative. When compiling executive intelligence reports, the model must be locked down to prevent it from wandering off into storytelling or hallucination.
3. **Dynamic Response Structure:** Generation outputs can return text either as a raw string tensor or a complex nested conversational dictionary list depending on packaging updates, causing traditional parsing loops to break.

---

## 🛠️ Technical Solution & Architecture
We solve this by refactoring our text processing suite to run under a unified generation pipeline, utilizing a lightweight local instruction model combined with explicit parameter constraints.

### Workflow:
1. **Context Loading:** Ingest the book chapters using our shared modular utilities `extract_book_text` and `chunk_text` to maintain a uniform data pipeline.
2. **Unified Task Abstraction:** Initialize the `text-generation` task running `HuggingFaceTB/SmolLM2-135M-Instruct` to handle multi-sentence compilation via prompt directives.
3. **Hyperparameter Hardening:** Apply strict generation parameters (`temperature=0.1`, `do_sample=False`) to remove model randomness and force a purely factual distillation of the context.
4. **Structural Defense Parsing:** Implement an `isinstance` runtime validation filter to safely handle, inspect, and normalize varying token responses into clean `pandas` rows.

---

# Assignment 5: The "What If?" Tactical Simulator 🔮⚽

## 📌 Problem Statement
**"How can we configure a local language model to write creative, logically consistent alternative-history football analysis based on specific book theories, without letting the text degenerate into endless repetitive loops?"**

### The Core Challenges:
1. **The Creative Control Paradox:** Unlike data retrieval tasks, simulation requires creative license. However, if unconstrained, the model will lose tactical logic and hallucinate historical impossibilities.
2. **The Repetition Loop Trap:** Small local language models often fall into deterministic patterns when writing longer strings, caught in a loop of repeating identical words or phrases over and over.
3. **Parameter Ingestion Conflicts:** Passing explicit decoding configurations (like temperature and sampling bounds) directly into chat pipeline layouts causes deprecation warnings and structural conflicts in modern Hugging Face versions.

---

## 🛠️ Technical Solution & Architecture
We solve this by shifting from deterministic reading to probabilistic generation. We isolate our configuration flags into a standalone dictionary argument wrapper and implement advanced decoding filters to keep creativity grounded.

### Workflow:
1. **Open-Domain Prompting:** Instead of reading text from a local `.epub` asset, tap into the model’s internal pre-trained vocabulary using conversational role directives.
2. **Keyword Kwargs Unpacking:** Bundle generation properties (`temperature`, `top_k`, `top_p`) into a clean configuration dictionary (`gen_kwargs`) and unpack them dynamically using Python’s double-asterisk (`**`) operator to maintain clean pipeline separation.
3. **Stochastic Sampling Execution:** Enable `do_sample=True` and tune the temperature to `0.75` to unlock imaginative linguistic variation.
4. **Token Repetition Suppression:** Implement a hard mathematical blockade (`no_repeat_ngram_size=3`) to intercept the token probability generation map and zero out repeating phrase paths before they can manifest.

---
