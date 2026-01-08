from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Dict, Any
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

class VectorDatabase:
    
    def __init__(self, persist_directory: str = "./chroma_db"):   
        self.vector_store = Chroma(
            collection_name="my_collection",
            embedding_function=embeddings,   
            persist_directory=persist_directory 
        )

    def delete_by_source(self, source: str) -> bool:
        try:
            # Get all documents with the specified source
            results = self.vector_store.get(
                where={"source": source},
                include=["documents"]
            )
            if results["documents"]:
                self.vector_store.delete(where={"source": source})
                logger.info(f"Deleted {len(results['documents'])} chunks from source: {source}")
                return True
            else:
                logger.info(f"No chunks found for source: {source}")
                return False
        except Exception as e:
            logger.error(f"Error deleting chunks by source: {e}")
            return False
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
            try:
                results = self.vector_store.get(include=["documents", "metadatas", "ids"])
                chunks = []
                
                for i in range(len(results['documents'])):
                    chunk = {
                        'id': results['ids'][i],
                        'content': results['documents'][i],
                        'metadata': results['metadatas'][i] if results['metadatas'] else {}
                    }
                    chunks.append(chunk)
                
                return chunks
            except Exception as e:
                logger.error(f"Error getting all chunks: {e}")
                return []
    
    def get_chunks_by_source(self, source: str) -> List[Document]:
        try:
            results = self.vector_store.get(
                where={"source": source},
                include=["documents", "metadatas"]
            )
            
            documents = []
            for i in range(len(results['documents'])):
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error getting chunks by source: {e}")
            return [] 

    def add_document_chunks(self, chunks: List[Document], source: str = None) -> None:
        try:
            # Ensure each chunk has proper metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Create a copy of the chunk to avoid modifying the original
                chunk_metadata = dict(chunk.metadata) if chunk.metadata else {}
                
                # Add source if provided and not already in metadata
                if source and "source" not in chunk_metadata:
                    chunk_metadata["source"] = source
                
                # Add chunk index for tracking
                chunk_metadata["chunk_index"] = i
                
                processed_chunk = Document(
                    page_content=chunk.page_content,
                    metadata=chunk_metadata
                )
                processed_chunks.append(processed_chunk)
            
            # Generate unique IDs for each chunk
            uuids = [str(uuid4()) for _ in range(len(processed_chunks))]
            
            # Add to vector store
            self.vector_store.add_documents(documents=processed_chunks, ids=uuids)
            logger.info(f"Added {len(processed_chunks)} chunks to database from source: {source}")
            
        except Exception as e:
            logger.error(f"Error adding chunks to database: {e}")
            raise

    def clear_database(self):
        try:
            self.vector_store.reset_collection()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []

    def get_sources(self):
        try:
            all_data = self.vector_store.get(include=["metadatas"])
            
            sources = set()
            for metadata in all_data["metadatas"]:
                source = metadata.get("source")
                if source:
                    sources.add(source)
            
            return list(sources) if sources else ["DATABASE IS EMPTY!"]
        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return ["ERROR RETRIEVING SOURCES"]
        
    def get_database_stats(self) -> Dict[str, Any]:
        try:
            all_data = self.vector_store.get(include=["metadatas"])
            total_chunks = len(all_data.get("metadatas", []))
            
            sources = {}
            for metadata in all_data.get("metadatas", []):
                source = metadata.get("source", "unknown")
                sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_chunks": total_chunks,
                "sources": sources,
                "unique_sources": len(sources)
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    