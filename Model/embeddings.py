import os
import json
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "final.jsonl"
INDEX_DIR = "faiss_index"

def load_products(filepath):
    products = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                products.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {e}")
    return products

def build_documents(products):
    docs = []
    for product in products:
        content = f"{product.get('Name', '')}\n{product.get('Description', '')}"
        metadata = {
            "Product Link": product.get("URL", ""),
            "Duration": product.get("Duration", ""),
            "Job Levels": product.get("Job Levels", ""),
            "Test Type": product.get("Test Type", ""),
            "Remote Support": product.get("Remote Support", ""),
            "Adaptive/IRT": product.get("Adaptive/IRT", "")
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def main():
    print("Loading products...")
    products = load_products(INPUT_FILE)
    print(f"{len(products)} products loaded.")
    print("Building documents...")
    docs = build_documents(products)
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Saving vector store to '{INDEX_DIR}'...")
    vectorstore.save_local(INDEX_DIR)
    print("Vector store built and saved successfully.")

if __name__ == "__main__":
    main()
