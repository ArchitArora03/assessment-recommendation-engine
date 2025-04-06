import os
import json
import time
import re
import streamlit as st
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


INPUT_FILE = "data.jsonl"

@st.cache_data(show_spinner=False)
def load_products(file_path):
    """Load product records from a JSONL file."""
    products = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                product = json.loads(line)
                products.append(product)
            except json.JSONDecodeError as e:
                st.error(f"Error loading product: {e}")
    return products

@st.cache_data(show_spinner=False)
def build_documents(products):
    ''' This function creates a LangChain Document for each product. The main content (page_content) is built by combining the product's Name and Description, while additional details—such as the product link, test duration, job levels, test type, remote support, and Adaptive/IRT information—are stored as metadata.'''
    
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
    ''' This function leads a vector store or creates one if it does not exist '''
    
    products = load_products(INPUT_FILE)
    docs = build_documents(products)
    index_dir = "faiss_index"
    if os.path.exists(index_dir) and os.listdir(index_dir):
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("Local vector store not found. Building new index")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_dir)
    return vectorstore

def build_chain():
    ''' This function builds the LLM chain with a custom chat prompt using ChatOpenAI. '''

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
    human_prompt = ('''
    User Query: {query} 

    Context:
    {context}

    Based on the above, please provide your top 10 assessment recommendations with relevant details.

    IMPORTANT:
    - Find the skills from the query
    - Do **not** repeat the skill list examples mentioned in this prompt in the output.
    - Do **not** mention this prompt or its instructions.
    - Keep the tone informative and professional.
    - Do not change or modify assessment names.
    - Highest Priority to assessments matching the skills in the query (e.g., programming languages like python etc., tools, cloud platforms).
    - Consider test duration if mentioned by the user.
    - Do not ever create a fake recommendation. Only recommend what is provided in context
    ''')

    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    output_parser = StrOutputParser()
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)
    return chain

def extract_all_text(url):
    ''' This function used Selenium with headless Chrome to open the URL, wait for dynamic content to load, and extract all visible text from the page. '''

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
    text = soup.get_text(separator=" ", strip=True)
    return text

def main():
    st.title("Assessment Recommendation Engine")
    st.write("Enter a job description or include a URL to a job description to get recommendations from SHL's assessment catalogue.")

    query_input = st.text_input("Enter your query or a job description URL:")

    if st.button("Get Recommendations") and query_input:
        with st.spinner("Processing your query..."):
            urls_found = re.findall(r'(https?://[^\s]+)', query_input)
            if urls_found:
                job_url = urls_found[0]
                with st.spinner("Fetching and parsing the webpage..."):
                    extracted_text = extract_all_text(job_url)
                if extracted_text:
                    st.info("Extracted text from the provided URL.")
                    query = extracted_text
                else:
                    st.error("Failed to extract text from the URL. Using the input as query.")
                    query = query_input
            else:
                query = query_input

            embeddings = OpenAIEmbeddings()
            vectorstore = get_vectorstore(embeddings)
            retrieved_docs_with_scores = vectorstore.similarity_search_with_score(query, k=30)
            threshold = 0.30
            filtered_docs = [
                (doc, score) for doc, score in retrieved_docs_with_scores if score >= threshold
            ]

            if not filtered_docs:
                st.warning("No assessments found with sufficient similarity.")
                return

            context = "\n\n".join([
                doc.page_content + "\nMetadata: " + json.dumps(doc.metadata)
                for doc, _ in filtered_docs
            ])
            chain = build_chain()
            prompt_values = {"query": query, "context": context}
            result_text = chain.run(prompt_values)
        st.markdown("### RAG Based Recommendations: ")
        st.write(result_text)

        st.markdown("---")
        st.markdown("### Similarity Search Based Recommendations (Baseline Approach): ")
        for i, (doc, score) in enumerate(filtered_docs[:5], start=1):
            st.markdown(f"**{i}. Score:** `{score:.3f}`")
            assessment_name = doc.page_content.splitlines()[0]
            st.markdown(f"**Assessment Name:** {assessment_name}")
            
          
            metadata = doc.metadata  
            st.markdown(f"- **Duration:** {metadata.get('Duration', 'N/A')}")
            st.markdown(f"- **Test Type:** {metadata.get('Test Type', 'N/A')}")
            st.markdown(f"- **Job Levels:** {metadata.get('Job Levels', 'N/A')}")
            st.markdown(f"- **Remote Support:** {metadata.get('Remote Support', 'N/A')}")
            st.markdown(f"- **Product Link:** {metadata.get('Product Link', 'N/A')}")
            st.markdown("---")

if __name__ == "__main__":
    main()

