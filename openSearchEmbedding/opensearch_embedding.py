from typing import List
from langchain_core.embeddings import Embeddings

# ---------------------------------
# Local OpenSearch Embedding Module
# ---------------------------------


class OpenSearchEmbeddings(Embeddings):
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        """
        return [self.embedding_function(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a single query.
        """
        return self.embedding_function(text)