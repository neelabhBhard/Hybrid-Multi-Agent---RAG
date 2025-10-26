"""
Embedding Service for RAG System
Provides proper vector embeddings using sentence-transformers
"""

import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:
    """Service for generating high-quality text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Default for all-MiniLM-L6-v2
        
        print(f" Initializing embedding service with default dimension: {self.embedding_dimension}")
        
        # Try to load the model
        try:
            print(f" Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"✓ Embedding service initialized with model: {model_name}")
            print(f"  Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            print(f" Warning: Could not load {model_name}, using fallback embeddings")
            print(f"  Error: {e}")
            self.model = None
            print(f"  Fallback embedding dimension: {self.embedding_dimension}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self.embedding_dimension
        
        try:
            if self.model:
                # Use the proper embedding model
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            else:
                # Fallback to simple hash-based embedding
                return self._generate_fallback_embedding(text)
                
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return self._generate_fallback_embedding(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (more efficient)
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            if self.model:
                # Use batch processing for efficiency
                embeddings = self.model.encode(texts, convert_to_tensor=False)
                return embeddings.tolist()
            else:
                # Fallback to individual processing
                return [self._generate_fallback_embedding(text) for text in texts]
                
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return [self._generate_fallback_embedding(text) for text in texts]
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding method when the model is not available"""
        # Simple hash-based embedding (not semantic, but provides consistent vectors)
        hash_val = hash(text) % 1000000
        
        # Generate a vector of the required dimension
        embedding = []
        for i in range(self.embedding_dimension):
            # Use hash + position to generate pseudo-random but consistent values
            val = (hash_val + i * 31) % 1000000
            embedding.append((val % 200 - 100) / 100.0)  # Values between -1 and 1
        
        return embedding
    
    def get_model_info(self) -> dict:
        """Get information about the current embedding model"""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "model_loaded": self.model is not None,
            "fallback_mode": self.model is None
        }
    
    def change_model(self, new_model_name: str) -> bool:
        """
        Change to a different embedding model
        
        Args:
            new_model_name: Name of the new model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            new_model = SentenceTransformer(new_model_name)
            new_dimension = new_model.get_sentence_embedding_dimension()
            
            self.model = new_model
            self.model_name = new_model_name
            self.embedding_dimension = new_dimension
            
            print(f" Successfully changed to model: {new_model_name}")
            print(f" New embedding dimension: {new_dimension}")
            return True
            
        except Exception as e:
            print(f" Error changing model to {new_model_name}: {e}")
            return False
