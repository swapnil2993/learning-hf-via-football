# 📘 Assignment 3: QA Search Engine (Simple Notes)

## 🧠 Goal

Build a system that can **answer questions from a full book**.

---

## 🚧 Problems

- Models cannot read full book (token limit)
- Answers can break between chunks
- Running on all text is slow

---

## 🛠️ What I Built

```
Book → Split into chunks → Find relevant chunk → Generate answer
```

---

## ⚙️ Key Ideas

- **Chunking**
  Split book into small parts (600 words + overlap)

- **Keyword Filter**
  Pick chunk that matches question words

- **AI Model**
  Generate answer from that chunk

---

## ⚠️ Limitations

- Picks only **first matching chunk** (may miss better answer)
- Uses **keyword search** (no deep understanding)
- Small model → answers may be weak or wrong
- Not true QA (it **generates**, not extracts)

---

## 🧠 What I Learned

- AI needs **good data + smart retrieval**
- Chunking is very important
- Better retrieval = better answers
- Trade-off between **speed and accuracy**
