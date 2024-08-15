import os
import pickle
try:
    import torch
    print(f"Torch version: {torch.__version__}")
except ImportError as e:
    print(f"Failed to import torch: {e}")
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text

# Text Extraction Function
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        bible_text = {}
        current_verse = None
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if ':' in line:
                parts = line.split(' ', 1)
                if len(parts) > 1 and ':' in parts[0]:
                    verse_key = parts[0].strip()
                    verse_text = parts[1].strip()
                    current_verse = verse_key
                    bible_text[current_verse] = verse_text
                elif current_verse:
                    bible_text[current_verse] += " " + line
        if not bible_text:
            print("Warning: No text extracted from the PDF.")
        return bible_text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

# Indexing Function
def create_index(bible_text):
    schema = Schema(verse=ID(stored=True), content=TEXT(stored=True))
    if not os.path.exists("indexdir"):
        os.mkdir("indexdir")
    ix = create_in("indexdir", schema)
    writer = ix.writer()

    for verse, content in bible_text.items():
        writer.add_document(verse=verse, content=content)

    writer.commit()

# Generate embeddings for semantic search
def generate_embeddings(corpus):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(corpus, convert_to_tensor=True)
    return embeddings

def main():
    # Extract text from PDF
    bible_text = extract_text_from_pdf("bible-old-testament.pdf")

    if bible_text is None or len(bible_text) == 0:
        print("Error: No text extracted. Check the PDF file.")
        return

    # Index the text
    create_index(bible_text)

    # Generate embeddings
    corpus = list(bible_text.values())
    embeddings = generate_embeddings(corpus)

    # Save embeddings and corpus for later use
    with open("corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
        print("corpus.pkl file created successfully.")

    torch.save(embeddings, "embeddings.pt")
    print("embeddings.pt file created successfully.")

if __name__ == "__main__":
    main()
