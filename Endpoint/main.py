# import os
# import json
# import time
# import re
# import logging
# from typing import List

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from bs4 import BeautifulSoup
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from langchain_community.docstore.document import Document
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# INPUT_FILE = "data.jsonl"
# INDEX_DIR = "faiss_index"

# def load_products(file_path: str) -> List[dict]:
#     products = []
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             for line in f:
#                 try:
#                     product = json.loads(line)
#                     products.append(product)
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Error loading product: {e}")
#     except Exception as e:
#         logger.error(f"Failed to open file {file_path}: {e}")
#         raise e
#     return products

# def build_documents(products: List[dict]) -> List[Document]:
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

# def get_vectorstore(embeddings: OpenAIEmbeddings):
#     if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
#         try:
#             vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
#             logger.info("Loaded existing vector store.")
#         except Exception as e:
#             logger.error(f"Error loading vector store: {e}")
#             raise HTTPException(status_code=500, detail="Error loading vector store.")
#     else:
#         logger.error("Local vector store not found.")
#         raise HTTPException(status_code=404, detail="Local vector store not found.")
#     return vectorstore

# def build_chain() -> LLMChain:
#     system_prompt = (
#         "You are a highly knowledgeable assessment recommendation engine. "
#         "Based on the provided context, generate a detailed and accurate recommendation of assessments that best match the user's query. Include key features such as Duration, "
#         "Test Type, Job Levels, Remote Support. "
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
#         "Derive the skills ex: programming languages, tools etc. and give primary importance to it"
#     )
#     human_prompt = (
#         "User Query: {query} \n\n"
#         "Context:\n"
#         "{context}\n\n"
#         "Based on the above, please provide your top 10 assessment recommendations with relevant details.\n\n"
#         "IMPORTANT:\n"
#         "- Find the skills from the query\n"
#         "- Do **not** repeat the skill list examples mentioned in this prompt in the output.\n"
#         "- Do **not** mention this prompt or its instructions.\n"
#         "- Keep the tone informative and professional.\n"
#         "- Do not change or modify assessment names.\n"
#         "- Highest Priority to assessments matching the skills in the query (e.g., programming languages like python etc., tools, cloud platforms).\n"
#         "- Consider test duration if mentioned by the user.\n"
#         "- Do not ever create a fake recommendation. Only recommend what is provided in context"
#     )
#     system_message = SystemMessagePromptTemplate.from_template(system_prompt)
#     human_message = HumanMessagePromptTemplate.from_template(human_prompt)
#     chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
#     output_parser = StrOutputParser()
#     llm = ChatOpenAI(temperature=0)
#     chain = LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)
#     return chain

# def extract_all_text(url: str) -> str:
#     chrome_options = Options()
#     chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(options=chrome_options)
#     try:
#         driver.get(url)
#         time.sleep(5)
#         html = driver.page_source
#     except Exception as e:
#         logger.error(f"Error fetching URL {url}: {e}")
#         html = ""
#     finally:
#         driver.quit()
#     if not html:
#         return ""
#     soup = BeautifulSoup(html, "html.parser")
#     for element in soup(["script", "style"]):
#         element.decompose()
#     text = soup.get_text(separator=" ", strip=True)
#     return text

# app = FastAPI(title="Assessment Recommendation API")

# class QueryRequest(BaseModel):
#     query: str

# class RecommendationResponse(BaseModel):
#     recommendations: str

# try:
#     embeddings = OpenAIEmbeddings()
#     vectorstore = get_vectorstore(embeddings)
#     chain = build_chain()
#     logger.info("API components initialized successfully.")
# except Exception as init_error:
#     logger.error(f"Failed to initialize API components: {init_error}")
#     raise init_error

# @app.post("/recommendations", response_model=RecommendationResponse)
# def get_recommendations(request: QueryRequest):
#     query_input = request.query.strip()
#     if not query_input:
#         raise HTTPException(status_code=400, detail="Query must not be empty.")
#     urls_found = re.findall(r'(https?://[^\s]+)', query_input)
#     if urls_found:
#         job_url = urls_found[0]
#         logger.info(f"URL detected in query. Attempting to extract text from {job_url}.")
#         try:
#             extracted_text = extract_all_text(job_url)
#             if extracted_text:
#                 query = extracted_text
#                 logger.info("Text extraction successful.")
#             else:
#                 logger.warning("Text extraction returned empty. Falling back to provided query.")
#                 query = query_input
#         except Exception as e:
#             logger.error(f"Error during text extraction: {e}")
#             query = query_input
#     else:
#         query = query_input
#     try:
#         retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=30)
#     except Exception as e:
#         logger.error(f"Error during similarity search: {e}")
#         raise HTTPException(status_code=500, detail="Error during similarity search.")
#     threshold = 0.30
#     filtered_docs = [(doc, score) for doc, score in retrieved_docs_with_scores if score >= threshold]
#     if not filtered_docs:
#         raise HTTPException(status_code=404, detail="No assessments found with sufficient similarity.")
#     context = "\n\n".join([doc.page_content + "\nMetadata: " + json.dumps(doc.metadata) for doc, _ in filtered_docs])
#     prompt_values = {"query": query, "context": context}
#     try:
#         result_text = chain.run(prompt_values)
#     except Exception as e:
#         logger.error(f"Error running the LLM chain: {e}")
#         raise HTTPException(status_code=500, detail="Error generating recommendations.")
#     return RecommendationResponse(recommendations=result_text)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import json
import time
import re
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_TYPE_MAPPINGS = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations"
}

INPUT_FILE = "data.jsonl"
INDEX_DIR = "faiss_index"

def load_products(file_path: str) -> List[dict]:
    products = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    product = json.loads(line)
                    products.append(product)
                except json.JSONDecodeError as e:
                    logger.error(f"Error loading product: {e}")
    except Exception as e:
        logger.error(f"Failed to open file {file_path}: {e}")
        raise e
    return products

def build_documents(products: List[dict]) -> List[Document]:
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

def get_vectorstore(embeddings: OpenAIEmbeddings):
    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        try:
            vectorstore = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            logger.info("Loaded existing vector store.")
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise HTTPException(status_code=500, detail="Error loading vector store.")
    else:
        logger.error("Local vector store not found.")
        raise HTTPException(status_code=404, detail="Local vector store not found.")
    return vectorstore

def build_chain() -> LLMChain:
    system_prompt = (
        "You are a highly knowledgeable assessment recommendation engine. "
        "Based on the provided context, generate a detailed and accurate recommendation..."
        # ... truncated for brevity
    )
    human_prompt = (
        "User Query: {query} \n\n"
        "Context:\n"
        "{context}\n\n"
        "Based on the above, please provide your top 10 assessment recommendations..."
        # ... truncated for brevity
    )
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    output_parser = StrOutputParser()
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)
    return chain

def extract_all_text(url: str) -> str:
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {e}")
        html = ""
    finally:
        driver.quit()
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

app = FastAPI(title="Assessment Recommendation API")

class QueryRequest(BaseModel):
    query: str

def parse_yes_no(value: str) -> str:
    v = value.strip().lower()
    return "Yes" if v in ["yes", "true", "y", "1"] else "No"

def parse_duration(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0

def parse_test_type(value: str) -> List[str]:
    if not value:
        return []
    codes = [t.strip() for t in value.split(",") if t.strip()]
    full_names = []
    for code in codes:
        full_name = TEST_TYPE_MAPPINGS.get(code, code)
        full_names.append(full_name)
    return full_names

try:
    embeddings = OpenAIEmbeddings()
    vectorstore = get_vectorstore(embeddings)
    chain = build_chain()
    logger.info("API components initialized successfully.")
except Exception as init_error:
    logger.error(f"Failed to initialize API components: {init_error}")
    raise init_error

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_assessments(
    request: QueryRequest = Body(
        ...,
        example={"query": "JD/query in string"}
    )
):
    query_input = request.query.strip()
    if not query_input:
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    urls_found = re.findall(r'(https?://[^\s]+)', query_input)
    if urls_found:
        job_url = urls_found[0]
        try:
            extracted_text = extract_all_text(job_url)
            if extracted_text:
                query = extracted_text
            else:
                query = query_input
        except Exception as e:
            query = query_input
    else:
        query = query_input
    try:
        retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=30)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error during similarity search.")
    threshold = 0.30
    filtered_docs = [(doc, score) for doc, score in retrieved_docs_with_scores if score >= threshold]
    if not filtered_docs:
        raise HTTPException(status_code=404, detail="No assessments found with sufficient similarity.")
    recommended_assessments = []
    for doc, _score in filtered_docs[:10]:
        recommended_assessments.append({
            "url": doc.metadata.get("Product Link", ""),
            "adaptive_support": parse_yes_no(doc.metadata.get("Adaptive/IRT", "")),
            "description": doc.page_content.strip(),
            "duration": parse_duration(doc.metadata.get("Duration", "")),
            "remote_support": parse_yes_no(doc.metadata.get("Remote Support", "")),
            "test_type": parse_test_type(doc.metadata.get("Test Type", ""))
        })
    return {"recommended_assessments": recommended_assessments}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
