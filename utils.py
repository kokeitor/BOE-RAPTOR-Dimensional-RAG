
import os
import streamlit as st

import chromadb
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings 


FILE_LIST = "archivos.txt"
INDEX_NAME = 'taller'

chroma_client = chromadb.HttpClient(host='localhost', port=8000)

def save_name_files(path, new_files):

    old_files = load_name_files(path)

    with open(path, "a") as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + "\n")
                old_files.append(item)
    
    return old_files


def load_name_files(path):

    archivos = []
    with open(path, "r") as file:
        for line in file:
            archivos.append(line.strip())

    return archivos


def clean_files(path):
    with open(path, "w") as file:
        pass
    chroma_client.delete_collection(name=INDEX_NAME)
    collection  = chroma_client.create_collection(
        name=INDEX_NAME,
        metadata={"hnsw:space": "l2"} # l2 is the default
    )

    return True


def text_to_chromadb(pdf):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, "wb") as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    doc_object = loader.load() 
    # los document objects tienen como atributos: .page_content [alamcena elk contenido paginas de pdf en tipo str] -- .metadata [dict con keys : "source" value: (ruta del  pdf absoluta), "page" etc]
    """ 
    # Prueba para cconocer mejor atributos de objeto document    
    with open('prueba.txt', "w") as file:
        for doc in doc_object:
            file.write(doc.page_content)
    """

    with st.spinner(f'Creando embedding fichero: {pdf.name}'):
        create_embeddings(pdf.name, doc_object)
    st.success('Embedding creado!', icon="✅")
    return True


def create_embeddings(file_name, doc_object):
    print(f"Creando embeddings del archivo: {file_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=10,
        length_function=len
        )        
    
    chunks = text_splitter.split_documents(doc_object)
    # Realmente este objeto llamado chunks son documents objects cerados a partir del document object padre pasado copmo argumento 
    # al metodo .split_documents() del objeto de la clase RecursiveCharacterTextSplitter. Este objeto te genera esos "sub doc objects" o chunks en funcion
    # de lo que le hayas pasado al constructor en la instancia de la clase RecursiveCharacterTextSplitter.
    print("chunks : ", chunks)
    print("chunks size: ", len(chunks))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    
    # from_documents es un classmethod de clase Chroma que es un modulo de langchain
    # este classmethod equivale a create_colletion de libreria chromadb solo que generalizado (p ejemplo: le pasa el chroma client y el nombre de la collection a la vez )
    # docstring classmethod: Create a Chroma vectorstore from a list of documents
    _chroma_obj_db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,   
        client = chroma_client,
        collection_name=INDEX_NAME)
    print("Colection info : ", _chroma_obj_db.get().keys())
    print("Colection info ids len : ", len(_chroma_obj_db.get()["ids"]))
    print("Colection info : ", _chroma_obj_db.get()["documents"])
    

    return True
