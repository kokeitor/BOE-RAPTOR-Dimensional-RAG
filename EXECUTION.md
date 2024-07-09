(env) C:\Users\Jorge\Desktop\MASTER_IA\TFM\proyecto\src>streamlit run --server.port 8052 streamlit_app.py

# DOCKER : 
docker pull chromadb/chroma
docker run  -p 8000:8000 --name chroma chromadb/chroma
buscador web local : http://localhost:8000/api/v1


# Comandos apara ejecutar la herramienta

1. navegar hasta Root proyecto
2. Configurar herramienta de download [configuracion json]:
    2.1 Configurar fecha inicial y fin BOE
    2.2 Configurar ruta de guardado (recomendado dejar como ./data)
3. ejecutar : python src/main_download.py

4. Configurar herramienta de etl [configuracion json]:
    4.1 parametros de configuracion ... 
5. ejecutar : python src/main_etl.py