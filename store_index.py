from src.helper import load_pdf,text_split,download_hugging_face_embedding
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')


extracted_data=load_pdf("Data/")
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embedding()


os.environ["PINECONE_API_KEY"] = "PINECONE_API_KEY"
index_name =pinecone.Index("medicalchatbot",host="https://medicalchatbot-2whmgb2.svc.aped-4627-b74a.pinecone.io")


from sentence_transformers import SentenceTransformer
import numpy as np
import pinecone
# Create vector embeddings for your text chunks in batches
def batch_encode(texts, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

# Initialize the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Get the text contents
text_contents = [t.page_content for t in text_chunks]

# Generate embeddings in batches
embeddings = batch_encode(text_contents, model)
embeddings = np.array(embeddings).tolist()

# Prepare data for upsert with metadata
data_to_upsert = [(str(i), embedding, {"text": text_contents[i]}) for i, embedding in enumerate(embeddings)]

# Define a function to upsert data in batches
def upsert_in_batches(index, data, batch_size=100):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        index.upsert(vectors=batch)

# Upsert data into Pinecone in smaller batches
upsert_in_batches(index, data_to_upsert)

print("Vector embeddings have been stored in Pinecone.")