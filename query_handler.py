from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\bPage\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bFax:.*?(?=\s|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\bFigure\s*\d+.*?(?=\s[A-Z]|\s\d+|$)', '', text, flags=re.IGNORECASE)
    return text.strip()

def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db_chroma", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    try:
        docs = db.get()
        print(f"\n✅ RAG chatbot ready. {len(docs['documents'])} chunks loaded.")
    except:
        print("\n⚠️ Could not count documents. But RAG is initialized.")

    print("\n💬 Type your question below (type 'exit' or 'quit' to stop):")
    print("💡 Example: What are the symptoms of heart disease?")

    while True:
        query = input("\n🧠 Question: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break

        try:
            docs = retriever.get_relevant_documents(query)
            if docs:
                for i, doc in enumerate(docs, start=1):
                    content_preview = clean_text(doc.page_content.strip())[:350]
                    source = doc.metadata.get('source', 'Unknown')
                    print(f"\n{i}. 📄 {content_preview}...")
                    print(f"   📁 Source: {source}")
            else:
                print("\n❌ No relevant information found.")
        except Exception as e:
            print(f"\n🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()
