from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import requests
import os
import chromadb
import uuid
import chromadb


client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection(name="pdf_embeddings")
app = Flask(__name__)

OLLAMA_EMBED_MODEL = "nomic-embed-text"  # Or your preferred embedding model
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_embedding_from_ollama(text, model=OLLAMA_EMBED_MODEL):
    payload = {
        "model": model,
        "prompt": text
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        raise Exception(f"Ollama error: {response.status_code}, {response.text}")

def add_embedding_to_chroma(text_chunk, embedding):
    collection.add(
        documents=[text_chunk],
        embeddings=[embedding],
        ids=[str(uuid.uuid4())]
    )

def query_chroma(query, top_k=3):
    query_embedding = get_embedding_from_ollama(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    # Filter out results with poor similarity
    threshold = 300
    filtered_docs = [
        doc for doc, dist in zip(results["documents"][0], results["distances"][0])
        if dist < threshold
    ]

    return filtered_docs or [''] # List of top_k chunks

def get_context(query):
    context_chunks = query_chroma(query)
    if context_chunks is not ['']:
      context = "\n\n".join(context_chunks)
    return context

def generate_answer(query):
    context = get_context(query)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "aya-expanse:8b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def split_text_into_chunks(text, max_tokens=500):
    import textwrap
    return textwrap.wrap(text, max_tokens)

@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.pdf'):
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)

        try:
            text = extract_text_from_pdf(temp_path)
            if not text.strip():
                return jsonify({'error': 'No text extracted from PDF'}), 400

            embedding = get_embedding_from_ollama(text)

            for chunk in split_text_into_chunks(text):
                embedding = get_embedding_from_ollama(chunk)
                add_embedding_to_chroma(chunk, embedding)

            return jsonify({'embedding': embedding})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDF is allowed'}), 400

@app.route('/reset-pdf-embeddings', methods=['DELETE'])
def reset_pdf_embeddings():
    collection = client.get_or_create_collection(name="pdf_embeddings")
    # Delete all embeddings from the collection
    collection.delete()
    return jsonify({'success': 'deleted all the embeddings successfully'})

@app.route('/query-agent', methods=['GET'])
def query_agent():
    query = request.form['query']
    answer = generate_answer(query)
    return jsonify({'answer': answer})

@app.route('/get-rag', methods=['GET'])
def get_rag():
    query = request.form['query']
    context = get_context(query)
    return jsonify({'context': context})

if __name__ == '__main__':
    app.run(debug=True)
