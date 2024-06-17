from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
import chromadb
from src.package.module.llm import EMBEDDING_MODEL,EMBEDDING_MODEL_GPT4
from typing import List,Tuple

INDEX_NAME = "llama3"

## RETRIEVER FUNCTION 
def docs_from_retriver(question :str, retriever):
    try: 
        return retriever.invoke(question)
    except Exception as e:
        print(f"{e}")

    try: 
        return retriever.invoke(question)
    except Exception as e:
        print(f"{e}")



### Wrapperv to try and get info for db
def try_db(func : callable = None):
    
    def wrapper(*args, **kwargs):
        
        ## Verify the storage inside chroma database
        chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone = func(*args, **kwargs)
        
        num = 2
        try: 
            for id in chroma_vectorstore.get()["ids"]:
                if num > 0:
                    print(chroma_vectorstore.get(include=['embeddings', 'documents', 'metadatas']))
                    num -= 1
                    
            # Prueba sobre db usando el retriever
            query = "La duración total de las enseñanzas en ciclos de grado medio"
            response = retriever_chroma.invoke(query)
            print("Number of embeddings retrieved : ", len(response))
            try:
                print("Best cosine similarity : ", response[0].page_content)
            except Exception as e:
                print(f"{e}")

        except NameError as e:
            print(f"{e}")


        try:
            # nota : Instanciar clase Chroma crea un objeto equivalnete a chroma_client de la libreria chromadb pero usando libreria langchain  
            print("Collection info : ", chroma_vectorstore.get().keys())
            print("Collection info ids len : ", (chroma_vectorstore.get()["ids"]))
            print("Collection docs : ", chroma_vectorstore.get()["documents"])
            try:
                print("Collection embeddings (1st comp of first embedding) : ", chroma_vectorstore.get(include = ['embeddings'])["embeddings"][0][0])
                print("LEN OF COLLECTION EMBEDDINGS: ", len(chroma_vectorstore.get(include = ['embeddings'])["embeddings"][0]))
            except Exception as e:
                print(f"{e}")
        except NameError as e:
            print(f"{e}")
        
        # Prueba sobre pinecone db usando el retriever
        try:
            query = "La duración total de las enseñanzas en ciclos de grado medio"
            response = retriever_pinecone.invoke(query)
            print("Number of embeddings retrieved :", len(response))
            print("Best cosine similarity :\n", response[0].page_content)
        except Exception as e:
            print(f"{e}")
            
        return chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone
    return wrapper



### Db conexion
@try_db
def db_conexion() -> Tuple:
    try : 
        # Conexion to ChromaDB running in container locally
        chroma_client = chromadb.HttpClient(host='localhost', port=8000)
    except ValueError as e:
        print(f"Not posible to connect to CHROMA DB: \n\t//Exception message : {e}")

    # Delete index if already exists
    try : 
        chroma_client.delete_collection(name=INDEX_NAME)
        print(f"CHROMA DB collection with name : {INDEX_NAME} deleted")
    except:
        print(f"No CHROMA DB collection with name {INDEX_NAME}")

    # Initialize a collection inside the vectorDB from documents chunks
    
    # CHROMA DB
    try:
        chroma_vectorstore = Chroma(
                                    embedding_function = EMBEDDING_MODEL,   
                                    client = chroma_client,
                                    collection_name=INDEX_NAME,
                                    collection_metadata = {"hnsw:space": "cosine"} # dict o [deafult] None donde le puedes pasar metadata igual que se hace en el metodo 
                                                                # : chroma_client.create_collection en su argumento (que tambien es un dict)
                                                                # : "metadata" --- ejemplo metadata={"hnsw:space": "l2"} l2 is default
                                            )

        # Croma Retriever
        retriever_chroma = chroma_vectorstore.as_retriever(search_kwargs = {"k" : 3})
        print("Conexion to CHROMA DB vectorestore correct: \n\t//CHROMA vectorstore created\n\t//CHROMA Retriever created")
        print(chroma_client.get_collection(name=INDEX_NAME))
        
    except Exception as e :
        print(f"LOCAL CHROMA DB does not respond: \n\t//Exception message : {e}")



    # PINECONE DB
    try:
        pinecone_vectorstore = PineconeVectorStore.from_existing_index(
                                                                    index_name = INDEX_NAME, 
                                                                    embedding = EMBEDDING_MODEL
                                                                )
        retriever_pinecone = pinecone_vectorstore.as_retriever(search_kwargs = {"k" : 3})
        print("Conexion to Pinecone DB vectorestore correct: \n\t//Pinecone vectorstore created\n\t//Pinecone Retriever created")
    except Exception as e :
        print(f"Pinecone DB does not respond: \n\t//Exception message : {e}")
        
        
    try_db(chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone)
        
    return chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone


def try_db(func : callable = None):
    
    def wrapper(*args, **kwargs):
        
        ## Verify the storage inside chroma database
        chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone = func(*args, **kwargs)
        
        num = 2
        try: 
            for id in chroma_vectorstore.get()["ids"]:
                if num > 0:
                    print(chroma_vectorstore.get(include=['embeddings', 'documents', 'metadatas']))
                    num -= 1
                    
            # Prueba sobre db usando el retriever
            query = "La duración total de las enseñanzas en ciclos de grado medio"
            response = retriever_chroma.invoke(query)
            print("Number of embeddings retrieved : ", len(response))
            try:
                print("Best cosine similarity : ", response[0].page_content)
            except Exception as e:
                print(f"{e}")

        except NameError as e:
            print(f"{e}")


        try:
            # nota : Instanciar clase Chroma crea un objeto equivalnete a chroma_client de la libreria chromadb pero usando libreria langchain  
            print("Collection info : ", chroma_vectorstore.get().keys())
            print("Collection info ids len : ", (chroma_vectorstore.get()["ids"]))
            print("Collection docs : ", chroma_vectorstore.get()["documents"])
            try:
                print("Collection embeddings (1st comp of first embedding) : ", chroma_vectorstore.get(include = ['embeddings'])["embeddings"][0][0])
                print("LEN OF COLLECTION EMBEDDINGS: ", len(chroma_vectorstore.get(include = ['embeddings'])["embeddings"][0]))
            except Exception as e:
                print(f"{e}")
        except NameError as e:
            print(f"{e}")
        
        # Prueba sobre pinecone db usando el retriever
        try:
            query = "La duración total de las enseñanzas en ciclos de grado medio"
            response = retriever_pinecone.invoke(query)
            print("Number of embeddings retrieved :", len(response))
            print("Best cosine similarity :\n", response[0].page_content)
        except Exception as e:
            print(f"{e}")
            
        return chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone
    return wrapper

    
