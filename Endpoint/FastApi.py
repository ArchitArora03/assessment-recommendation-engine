# import os
# import json
# import time
# import re
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from langchain.docstore.document import Document
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from dotenv import load_dotenv
# load_dotenv()  # Loads environment variables including OPENAI_API_KEY

# # Set the path to your JSONL file with product records.
# INPUT_FILE = "final.jsonl"  # Make sure this file is in your repo or accessible on the server.

# # Pydantic model for the incoming request.
# class QueryRequest(BaseModel):
#     query: str

# # Functions for loading data, building documents, and creating the vector store.
# def load_products(file_path):
#     products = []
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             try:
#                 product = json.loads(line)
#                 products.append(product)
#             except json.JSONDecodeError as e:
#                 print(f"Error loading product: {e}")
#     return products

# def build_documents(products):
#     docs = []
#     for product in products:
#         content = product.get("Name", "") + "\n" + product.get("Description", "")
#         metadata = {
#             "Product Link": product.get("URL", ""),
#             "Duration": product.get("Duration", ""),
#             "Job Levels": product.get("Job Levels", ""),
#             "Test Type": product.get("Test Type", ""),
#             "Remote Support": product.get("Remote Support", ""),
#             "Adaptive/IRT": product.get("Adaptive/IRT", "")
#         }
#         docs.append(Document(page_content=content, metadata=metadata))
#     return docs

# def get_vectorstore(embeddings):
#     products = load_products(INPUT_FILE)
#     docs = build_documents(products)
#     index_dir = "faiss_index"
#     if os.path.exists(index_dir) and os.listdir(index_dir):
#         vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
#         print("Loaded vector store from local storage.")
#     else:
#         print("Local vector store not found. Building new index...")
#         vectorstore = FAISS.from_documents(docs, embeddings)
#         vectorstore.save_local(index_dir)
#         print("Vector store built and saved locally.")
#     return vectorstore

# def build_chain():
#     system_prompt = (
#         "You are a highly knowledgeable assessment recommendation engine. "
#         "Based on the provided context, generate a detailed and accurate recommendation "
#         "of assessments that best match the user's query. Include key features such as Duration, "
#         "Test Type, Job Levels, Remote Support, and Adaptive/IRT support. "
#         "Format your answer clearly and concisely. Out of the 30 possible recommended assessments, "
#         "choose only the best 10 and include all details about them.\n\n"
#         "Test Type Codes:\n"
#         "A: Ability & Aptitude\n"
#         "B: Biodata & Situational Judgement\n"
#         "C: Competencies\n"
#         "D: Development & 360\n"
#         "E: Assessment Exercises\n"
#         "K: Knowledge & Skills\n"
#         "P: Personality & Behavior\n"
#         "S: Simulations\n"
#     )
#     human_prompt = (
#         "User Query: {query}\n\nContext:\n{context}\n\n"
#         "Based on the above, please provide your top 10 assessment recommendations with relevant details. "
#         "Do not change or modify assessment names. "
#         "Give importance to matching skill names in the query and assessment names, and consider test duration if mentioned."
#     )
#     system_message = SystemMessagePromptTemplate.from_template(system_prompt)
#     human_message = HumanMessagePromptTemplate.from_template(human_prompt)
#     chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
#     output_parser = StrOutputParser()
#     llm = ChatOpenAI(temperature=0)
#     chain = LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)
#     return chain

# # Pre-load embeddings, vectorstore, and chain to avoid recomputation.
# embeddings = OpenAIEmbeddings()
# vectorstore = get_vectorstore(embeddings)
# chain = build_chain()

# # Create the FastAPI app.
# app = FastAPI()

# @app.post("/recommend", response_model=dict)
# def recommend(query_request: QueryRequest):
#     query = query_request.query
#     # Retrieve the top 30 documents.
#     retrieved_docs = vectorstore.similarity_search(query, k=30)
#     context = "\n\n".join([
#         doc.page_content + "\nMetadata: " + json.dumps(doc.metadata)
#         for doc in retrieved_docs
#     ])
#     prompt_values = {"query": query, "context": context}
#     try:
#         result_text = chain.run(prompt_values)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     return {"result": result_text}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import os
import json
import time
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "data.jsonl"
INDEX_DIR = "faiss_index"
THRESHOLD = 0.30

app = FastAPI()

class QueryInput(BaseModel):
    query: str

def load_products(file_path):
    products = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                product = json.loads(line)
                products.append(product)
            except json.JSONDecodeError:
                continue
    return products

def build_documents(products):
    docs = []
    for product in products:
        content = product.get("Name", "") + "\n" + product.get("Description", "")
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

def get_vectorstore(embeddings):
    products = load_products(INPUT_FILE)
    docs = build_documents(products)
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(INDEX_DIR)
        return vectorstore

def extract_all_text(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(5)
    html = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return soup.get_text(separator=" ", strip=True)

def build_chain():
    system_prompt = (
        "You are a highly knowledgeable assessment recommendation engine. "
        "Based on the provided context, generate a detailed and accurate recommendation of assessments that best match the user's query. Include key features such as Duration, "
        "Test Type, Job Levels, Remote Support"
        "Format your answer clearly and concisely. Out of the 30 possible recommended assessments, "
        "choose only the best 10 and include all details about them.\n\n"
        "Test Type Codes:\n"
        "A: Ability & Aptitude\n"
        "B: Biodata & Situational Judgement\n"
        "C: Competencies\n"
        "D: Development & 360\n"
        "E: Assessment Exercises\n"
        "K: Knowledge & Skills\n"
        "P: Personality & Behavior\n"
        "S: Simulations\n"
        "Derive the skills ex: programming languages, tools etc. and give primary importance to it"
    )
    human_prompt = (
        "User Query: {query} \n\n"
        "Context:\n{context}\n\n"
        "Based on the above, please provide your top 10 assessment recommendations with relevant details.\n\n"
        "IMPORTANT:\n"
        "- Find the skills from the query\n"
        "- Do **not** repeat the skill list examples mentioned in this prompt in the output.\n"
        "- Do **not** mention this prompt or its instructions.\n"
        "- Keep the tone informative and professional.\n"
        "- Do not change or modify assessment names.\n"
        "- Highest Priority to assessments matching the skills in the query (e.g., programming languages like python etc., tools, cloud platforms).\n"
        "- Consider test duration if mentioned by the user.\n"
        "- Do not ever create a fake recommendation. Only recommend what is provided in context"
    )
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    output_parser = StrOutputParser()
    llm = ChatOpenAI(temperature=0)
    return LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)

@app.post("/recommend")
async def recommend_assessments(payload: QueryInput):
    query_input = payload.query
    urls_found = re.findall(r'(https?://[^\s]+)', query_input)
    if urls_found:
        try:
            query = extract_all_text(urls_found[0]) or query_input
        except:
            query = query_input
    else:
        query = query_input

    embeddings = OpenAIEmbeddings()
    vectorstore = get_vectorstore(embeddings)
    retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=30)
    filtered_docs = [(doc, score) for doc, score in retrieved_docs_with_scores if score >= THRESHOLD]

    if not filtered_docs:
        return {"rag_recommendations": "", "similarity_recommendations": []}

    context = "\n\n".join([
        doc.page_content + "\nMetadata: " + json.dumps(doc.metadata)
        for doc, _ in filtered_docs
    ])

    chain = build_chain()
    result_text = chain.run({"query": query, "context": context})

    top_matches = []
    for doc, score in filtered_docs[:5]:
        metadata = doc.metadata
        top_matches.append({
            "Score": round(score, 3),
            "Assessment Name": doc.page_content.splitlines()[0],
            "Duration": metadata.get("Duration", ""),
            "Test Type": metadata.get("Test Type", ""),
            "Job Levels": metadata.get("Job Levels", ""),
            "Remote Support": metadata.get("Remote Support", ""),
            "Product Link": metadata.get("Product Link", "")
        })

    return {
        "rag_recommendations": result_text,
        "similarity_recommendations": top_matches
    }
