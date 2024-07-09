import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from ETL.utils import get_current_spanish_date_iso, setup_logging
from ETL.etl import Pipeline
from databases.google_sheets import GoogleSheet


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
    os.environ['PINECONE_COLLECTION_NAME'] = os.getenv('PINECONE_COLLECTION_NAME')
    os.environ['CHROMA_COLLECTION_NAME'] = os.getenv('CHROMA_COLLECTION_NAME')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['APP_MODE'] = os.getenv('APP_MODE')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
    os.environ['EMBEDDING_MODEL'] = os.getenv('EMBEDDING_MODEL')
    os.environ['EMBEDDING_MODEL_GPT4'] = os.getenv('EMBEDDING_MODEL_GPT4')
    os.environ['LOCAL_LLM'] = os.getenv('LOCAL_LLM')
    os.environ['LOCAL_LLM'] = os.getenv('GOOGLE_BBDD_FILE_NAME_CREDENTIALS')
    os.environ['GOOGLE_DOCUMENT_NAME'] = os.getenv('GOOGLE_DOCUMENT_NAME')
    os.environ['GOOGLE_SHEET_NAME'] = os.getenv('GOOGLE_SHEET_NAME')
 
        
    # Logger set up
    setup_logging()
    
    # BBDD google sheet
    GOOGLE_SECRETS_FILE_NAME = os.path.join('.secrets',os.getenv('GOOGLE_BBDD_FILE_NAME_CREDENTIALS')) # only for local performance
    GOOGLE_DOCUMENT_NAME = os.environ['GOOGLE_DOCUMENT_NAME']# google sheet document name
    GOOGLE_SHEET_NAME = os.environ['GOOGLE_SHEET_NAME'] # google sheet name
    logger.info(f"Secrets BBDD path : {GOOGLE_SECRETS_FILE_NAME=}")
    
    # Google Sheet database object
    # bbdd_credentials = st.secrets["google"]["google_secrets"] # Google api credentials [drive and google sheets as bddd]
    BBDD = GoogleSheet(credentials=GOOGLE_SECRETS_FILE_NAME, document=GOOGLE_DOCUMENT_NAME, sheet_name=GOOGLE_SHEET_NAME)
    
    ETL_CONFIG_PATH = os.path.join(os.path.abspath("./config/etl"),"etl.json")
    pipeline = Pipeline(config_path=ETL_CONFIG_PATH, database=BBDD)
    result = pipeline.run()

    text = """ En este apartado se valorará, en su caso, el grado reconocido como personal
    funcionario de carrera en otras Administraciones Públicas o en la Sociedad Estatal de
    Correos y Telégrafos, en el Cuerpo o Escala desde el que participa el funcionario o
    funcionaria de carrera, cuando se halle dentro del intervalo de niveles establecido en el
    artículo 71.1 del Real Decreto 364/1995, de 10 de marzo, para el subgrupo de titulación
    en el que se encuentra clasificado el mismo.

    En el supuesto de que el grado reconocido en el ámbito de otras Administraciones
    Públicas o en la Sociedad Estatal de Correos y Telégrafos exceda del máximo
    establecido en la Administración General del Estado, de acuerdo con el artículo 71 del
    Reglamento mencionado en el punto anterior, para el subgrupo de titulación a que
    pertenezca el funcionario o la funcionaria de carrera, deberá valorársele el grado máximo
    correspondiente al intervalo de niveles asignado a su subgrupo de titulación en la
    Administración General del Estado.

    El funcionario o la funcionaria de carrera que considere tener un grado personal
    consolidado, o que pueda ser consolidado durante el periodo de presentación de
    instancias, deberá recabar del órgano o unidad a que se refiere el apartado 1 de la Base
    Quinta, que dicha circunstancia quede expresamente reflejada en el anexo
    correspondiente al certificado de méritos (anexo II)."""

    """
    s = Splitter(
        embedding_model=EMBEDDING_MODEL,
        tokenizer_model=tokenizer_llama3,
        threshold=75,
        max_tokens=500,
        verbose=1,
        buffer_size=3,
        max_big_chunks=4,
        splitter_mode='CUSTOM',
        embedding_for_research='HG',
        score_threshold_for_research=0.82,
    )
    doc = Document(page_content=text, metadata={"hola": '1'})
    split_docs = s.invoke(docs=[doc])

    # Example of usage:
    pipeline = Pipeline(config_path='path_to_config.json')
    result = pipeline.run()
    
    """
    

if __name__ == '__main__':
    main()
