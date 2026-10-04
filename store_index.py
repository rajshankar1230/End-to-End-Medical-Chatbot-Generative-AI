import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("1. Extracting data from PDF...")
extracted_data = load_pdf_file(data='Data/')

print("2. Splitting text into chunks...")
text_chunks = text_split(extracted_data)

print("3. Initializing Hugging Face embeddings...")
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Check if index already exists to avoid 409 Conflict error
existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new Pinecone index: '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    )
    # Wait briefly for serverless index to be provisioned
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

print("4. Uploading embeddings to Pinecone (this may take 1-2 minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

print("Successfully indexed medical documents into Pinecone!")