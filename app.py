from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
import pinecone
from langchain_community.prompts import PromptTemplate
from langchain_community.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_community.vectorstores import Pinecone
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader


app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

embeddings=download_hugging_face_embedding()

os.environ["PINECONE_API_KEY"] = "PINECONE_API_KEY"
index_name =pinecone.Index("medicalchatbot",host="https://medicalchatbot-2whmgb2.svc.aped-4627-b74a.pinecone.io")

docsearch=Pinecone.from_existing_index(index_name,embeddings)


PROMPT=PromptTemplate(template=prompt_template,input_variables=["context","question"])

chain_type_kwargs={"prompt": PROMPT}


llm=CTransformers(model=r"Model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriver=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template("chat.html")



if __name__== '__main__':
    app.run(debug=True)