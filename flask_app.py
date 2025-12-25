from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import re

app = Flask(__name__)
CORS(app)

def clean_text(text):
    # Clean and split into bullet points or short sentences
    text = re.sub(r'\([^)]*?\d{4}[^)]*?\)', '', text)  # remove citations
    text = re.sub(r'\([^)]*?\)', '', text)  # remove other brackets
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_points(text):
    # Try to split by numbered bullets or long sentences
    points = re.split(r'\s*\d+\.\s+', text)
    points = [p.strip() for p in points if len(p.strip()) > 30]  # keep non-empty decent lines
    return points

# Load vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db_chroma", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('question', '').strip()

        if not query:
            return jsonify({'answer': '⚠️ No question received.'})

        docs = retriever.get_relevant_documents(query)

        if not docs:
            return jsonify({'answer': '❌ Sorry, no relevant information found.'})

        all_points = []
        for doc in docs:
            content = clean_text(doc.page_content)
            points = split_into_points(content)
            all_points.extend(points)

        # Limit total points
        final_points = all_points[:5]  # Only show first 5 key points
        structured = "\n\n".join([f"{i+1}. {pt}" for i, pt in enumerate(final_points)])

        return jsonify({'answer': structured})

    except Exception as e:
        return jsonify({'answer': f'🚨 Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
