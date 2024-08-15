import os
import streamlit as st
import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        st.write(f"Package {package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.write(f"Package {package} installed successfully.")
        __import__(package)

# Check and install necessary packages
install_and_import("torch")
install_and_import("sentence_transformers")
install_and_import("whoosh")
install_and_import("openai")

# Now you can safely import these modules
from sentence_transformers import util, SentenceTransformer
import torch
import pickle
from openai import OpenAI

client = OpenAI()

# Semantic Search Function
def semantic_search(query, corpus, corpus_embeddings):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    query_embedding = model.encode(query, convert_to_tensor=True)
    search_results = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    results = []
    for result in search_results[0]:
        matched_text = corpus[result['corpus_id']]
        results.append(matched_text)
    return results

# OpenAI Function
def ask_openai(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert biblical scholar."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

# Main Function to Run Everything
def main():
    corpus_file = "corpus.pkl"
    embeddings_file = "embeddings.pt"

    # Streamlit App Title
    st.title("Bible Pocket Guide Chat")

    # Check if preprocessed files exist
    if not os.path.exists(corpus_file) or not os.path.exists(embeddings_file):
        st.error("Necessary files are missing. Please run the preprocessing script first.")
        return

    # Load precomputed data
    with open(corpus_file, "rb") as f:
        try:
            corpus = pickle.load(f)
        except EOFError:
            st.error("Error: Failed to load corpus.pkl. The file may be corrupted.")
            return

    corpus_embeddings = torch.load(embeddings_file)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar="📚"):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Capture user input
    user_input = st.chat_input("Ask any question about the Bible...")

    if user_input:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Perform semantic search
        search_results = semantic_search(user_input, corpus, corpus_embeddings)
        if search_results:
            context = "\n".join(search_results[:3])  # Use top 3 results as context
            answer = ask_openai(user_input, context)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No relevant sections found."})

        # Re-render the chat messages including the latest response
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message(message["role"], avatar="📚"):
                    st.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

if __name__ == "__main__":
    main()
