import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

print("Loaded Groq Key prefix:", GROQ_API_KEY[:8] if GROQ_API_KEY else "None")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

# Connect to the Pinecone vector index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Free, fast LLM via Groq (LLaMA 3.1 8B)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b",
    temperature=0.3,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\nContext:\n{context}"),
        ("human", "{input}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    print(f"User Query: {msg}")
    response = rag_chain.invoke(msg)
    print(f"Response: {response}")
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)