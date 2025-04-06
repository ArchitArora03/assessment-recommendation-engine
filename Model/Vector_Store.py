import json
from langchain.docstore.document import Document
from langchain.chains import LLMChain
import json
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
load_dotenv()

INPUT_FILE = "/Users/architarora/Desktop/Archit Arora/8th Semester/New Folder/assessment-recommendation-engine/Data/final.jsonl"

def load_products(file_path):
    """Load product records from a JSONL file."""
    products = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                product = json.loads(line)
                products.append(product)
            except json.JSONDecodeError as e:
                print("Error loading product:", e)
    return products

def build_documents(products):
    """
    Build LangChain Document objects.
    The page_content is a combination of the Name and Description.
    Metadata contains URL, Duration, Job Levels, Test Type, Remote Support, and Adaptive/IRT.
    """
    docs = []
    for product in products:
        content = product.get("Name", "") + "\n" + product.get("Description", "")
        metadata = {
            "URL": product.get("URL", ""),
            "Duration": product.get("Duration", ""),
            "Job Levels": product.get("Job Levels", ""),
            "Test Type": product.get("Test Type", ""),
            "Remote Support": product.get("Remote Support", ""),
            "Adaptive/IRT": product.get("Adaptive/IRT", "")
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def get_vectorstore(docs, embeddings):
    """
    Try to load the FAISS vector store from a local directory.
    If not found, create a new one from the provided documents and save it locally.
    """
    index_dir = "faiss_index"
    try:
        vectorstore = FAISS.load_local(index_dir, embeddings)
        print("Loaded vector store from local storage.")
    except Exception as e:
        print("Local vector store not found or failed to load. Building new index...")
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_dir)
        print("Vector store built and saved locally.")
    return vectorstore

def main():
    # Step 1: Load product data and build documents.
    products = load_products(INPUT_FILE)
    print(f"Loaded {len(products)} products from {INPUT_FILE}")

    docs = build_documents(products)

    # Step 2: Create (or load) the FAISS vector store using OpenAI embeddings.
    embeddings = OpenAIEmbeddings()
    vectorstore = get_vectorstore(docs, embeddings)

    # Retrieve the top 15 documents for the user query.
    query = input("Enter a job description or query: ")
    retrieved_docs = vectorstore.similarity_search(query, k=15)

    # Step 3: Build a custom chat prompt for Retrieval-Augmented Generation.
    system_prompt = ('''
        "You are a highly knowledgeable assessment recommendation engine. "
        "Based on the provided context, generate a detailed and accurate recommendation "
        "of assessments that best match the user's query. Include key features such as Duration, "
        "Test Type, Job Levels, Remote Support, and Adaptive/IRT support. "
        "Format your answer clearly and concisely. Out of the 15 possible recommended assessments, "
        "choose only the best 10 and include all details about them."
        ""Test Type Codes:\n"
        "A: Ability & Aptitude\n"
        "B: Biodata & Situational Judgement\n"
        "C: Competencies\n"
        "D: Development & 360\n"
        "E: Assessment Exercises\n"
        "K: Knowledge & Skills\n"
        "P: Personality & Behavior\n"
        "S: Simulations\n"'''
    )
    human_prompt = (
        "User Query: {query}\n\nContext:\n{context}\n\nBased on the above, please provide your top 10 assessment recommendations with relevant details. "
        "Give more importance to skills mentioned in the query (for example: SQL, Java, Python). "
        "Also, consider test duration if mentioned in the query."
    )
    system_message = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Concatenate retrieved documents into context.
    context = "\n\n".join([
        doc.page_content + "\nMetadata: " + json.dumps(doc.metadata)
        for doc in retrieved_docs
    ])

    # Prepare the prompt with the user's query and the retrieved context.
    prompt_values = {"query": query, "context": context}

    # Create an output parser to obtain plain text.
    output_parser = StrOutputParser()

    # Create the LLMChain using ChatOpenAI.
    llm = ChatOpenAI(temperature=0)
    chain = LLMChain(prompt=chat_prompt, llm=llm, output_parser=output_parser)

    # Generate the final recommendation result.
    result_text = chain.run(prompt_values)
    
    
    print(result_text)

if __name__ == "__main__":
    main()
