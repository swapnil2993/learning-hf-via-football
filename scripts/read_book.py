import os
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup


def extract_book_text(epub_path = None):
    if epub_path is None:
      BASE_DIR = os.path.dirname(os.path.abspath(__file__))
      epub_path = os.path.abspath(os.path.join(BASE_DIR, "..", "ebooks", "the-inverted-pyramid.epub"))
    print(f"📖 Reading book file: {os.path.basename(epub_path)}...")
    book = epub.read_epub(epub_path)
    full_text = []
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text().strip()
            if text:
                full_text.append(text)
    return " ".join(full_text)

def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     epub_file = os.path.abspath(os.path.join(BASE_DIR, "..", "ebooks", "the-inverted-pyramid.epub"))
#     book_text = extract_book_text(epub_file)
#     print(f"Extracted text length: {len(book_text)} characters")