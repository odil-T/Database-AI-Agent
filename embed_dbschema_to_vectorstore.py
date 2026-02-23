from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from config import TABLE_SCHEMAS_DIR
from chroma_vectorstore import get_vectorstore

vectorstore = get_vectorstore()

table_schema_documents = []

for filename in TABLE_SCHEMAS_DIR.iterdir():
    with open(filename, "r") as f:
        table_schema = f.read()
    table_schema_document = Document(page_content=table_schema, id=str(uuid4()))
    table_schema_documents.append(table_schema_document)

vectorstore.add_documents(table_schema_documents)
