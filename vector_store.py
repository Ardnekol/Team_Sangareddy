"""
Vector store module for storing and retrieving ticket embeddings.
Uses FAISS for efficient similarity search.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any


class TicketVectorStore:
    """Manages ticket embeddings and similarity search using FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'ticket_index.faiss'):
        """
        Initialize the vector store.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to save/load FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = index_path.replace('.faiss', '_metadata.pkl')
        
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []
        self.dimension = self.encoder.get_sentence_embedding_dimension()
    
    def build_index(self, ticket_texts: List[str], ticket_metadata: List[Dict[str, Any]]):
        """
        Build FAISS index from ticket texts.
        
        Args:
            ticket_texts: List of ticket text strings
            ticket_metadata: List of ticket metadata dictionaries
        """
        print(f"Encoding {len(ticket_texts)} tickets...")
        embeddings = self.encoder.encode(ticket_texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (Inner Product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        self.metadata = ticket_metadata
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        print(f"Index saved to {self.index_path}")
        print(f"Metadata saved to {self.metadata_path}")
    
    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            return False
        
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Index loaded with {self.index.ntotal} vectors")
        return True
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar tickets.
        
        Args:
            query: Query text string
            k: Number of results to return
            
        Returns:
            List of tuples (metadata, similarity_score)
        """
        if self.index is None:
            raise ValueError("Index not loaded. Call load_index() or build_index() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Return results with metadata
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(score)))
        
        return results

