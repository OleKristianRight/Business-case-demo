import pandas as pd
import numpy as np
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from dotenv import load_dotenv

# Last inn miljøvariabler
load_dotenv()

def create_embeddings_from_df(df):
    """
    Konverterer en DataFrame til embeddings og lager en FAISS indeks.
    
    Args:
        df (pd.DataFrame): DataFrame som skal konverteres til embeddings
        
    Returns:
        FAISS: FAISS indeks med embeddings
    """
    # Konverter DataFrame til tekst - bruk hele datasettet
    documents = []
    
    # Bruk hele datasettet (fjernet begrensning på 100 rader)
    for idx, row in df.iterrows():
        # Lag en kompakt tekstrepresentasjon
        text = ", ".join([f"{col}: {str(val)}" for col, val in row.items() if pd.notna(val)])
        documents.append(Document(page_content=text, metadata={"row_index": idx}))
    
    # Bruk større chunk-størrelse for å få med mer data per chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Økt fra 500 for å få med mer data
        chunk_overlap=100,  # Økt fra 50
        length_function=len
    )
    
    # Del opp dokumentene
    chunks = text_splitter.split_documents(documents)
    
    # Bruk alle chunks (fjernet begrensning på 50 chunks)
    print(f"Prosesserer {len(chunks)} chunks fra {len(df)} rader")
    
    # Opprett Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("target_url"),
        api_key=os.getenv("OPENAI_API_KEY"),
        chunk_size=100  # Økt fra 16 for raskere prosessering
    )
    
    # Opprett FAISS indeks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def query_data(vectorstore, query, k=50):
    """
    Søker i FAISS indeksen med en spørsmål.
    
    Args:
        vectorstore (FAISS): FAISS indeks
        query (str): Spørsmålet
        k (int): Antall resultater som skal returneres (økt til 50)
        
    Returns:
        list: Liste med relevante dokumenter
    """
    docs = vectorstore.similarity_search(query, k=k)
    return docs 