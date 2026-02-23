from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_DIR, CHROMA_VECTORSTORE_DIR


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=str(EMBEDDING_MODEL_DIR))
    vectorstore = Chroma(
        collection_name="citypharm_db_schema",
        embedding_function=embeddings,
        persist_directory=CHROMA_VECTORSTORE_DIR,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )
    return vectorstore
