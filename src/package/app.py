import streamlit as st
import os
from src.package.utils.utils_app import *
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_API_KEY = "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"

st.set_page_config('preguntaDOC KOKE')
st.header("Pregunta a tu PDF")


with st.sidebar:
    
    archivos = load_name_files(FILE_LIST)
    files_uploaded = st.file_uploader(
        "Carga tu archivo",
        type="pdf",
        accept_multiple_files=True
        )
    
    if st.button('Procesar'):
        for pdf in files_uploaded:
            #print("pdf type : ",type(pdf))
            if pdf is not None and pdf.name not in archivos:
                archivos.append(pdf.name)
                text_to_chromadb(pdf)

        archivos = save_name_files(FILE_LIST, archivos)
        
    if len(archivos) > 0:
        st.write("Archivos pdf procesados:")
        lista_documentos = st.empty()
        
        with lista_documentos.container():    
            #st.text("Has cargado los siguientes pdf : ") #menasaje opcional dentro del contain er que se borrara al pulsar boton borrar
            for arch in archivos:
                st.write(arch)
            if st.button('Borrar documentos'):
                archivos = []
                clean_files(FILE_LIST)
                lista_documentos.empty()

if archivos:
    user_question = st.text_input("Pregunta:")
    if user_question:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        # Conexion con db de chroma ya creada y en concreto a nuestra collection que alamacena : vectors(embbedings de nuestros chunks) y chunks
        _chroma_obj_db_2 = Chroma(
                                    client=chroma_client,
                                    collection_name=INDEX_NAME,
                                    embedding_function=embeddings,
                                    collection_metadata = None # dict o [deafult] None donde le puedes pasar metadata igual que se hace en el metodo 
                                                               # : chroma_client.create_collection en su argumento (que tambien es un dict)
                                                               # : "metadata" --- ejemplo metadata={"hnsw:space": "l2"}
                                )
        
        # metodo .similarity_search equivale a metodo .query (del objeto collection de chroma db)
        # retorna objeto document con (recordemos) tres atributos o propertys: page_content, metadata y type
        docs = _chroma_obj_db_2.similarity_search(query = user_question, k= 3)
        
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)

        st.write(respuesta)