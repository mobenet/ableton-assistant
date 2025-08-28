import os
from langchain_chroma import Chroma
from embeddings import E5Embeddings
vs = Chroma(collection_name=os.getenv("COLLECTION_NAME","ableton"),
            persist_directory=os.getenv("CHROMA_DIR","chroma_db"),
            embedding_function=E5Embeddings())
print("Docs:", vs._collection.count())