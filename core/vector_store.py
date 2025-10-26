"""
Vector store implementation using FAISS
Handles document embeddings storage and retrieval for RAG system
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from utils.text_chunker import TextChunk


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, 
                 index_path: str = "vector_index",
                 embedding_dimension: int = 384,  # Default for sentence-transformers
                 index_type: str = "flat",
                 force_clean_start: bool = False):
        """
        Initialize the vector store
        
        Args:
            index_path: Directory to store the FAISS index and metadata
            embedding_dimension: Dimension of the embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
            force_clean_start: If True, ignore existing files and start fresh
        """
        self.index_path = Path(index_path)
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.force_clean_start = force_clean_start
        
        # Create directory if it doesn't exist
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.chunk_metadata = {}
        self.document_mapping = {}
        
        # Load existing index if available (unless forcing clean start)
        if self.force_clean_start:
            self._force_clean_start()
        else:
            self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create a new one"""
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            try:
                # Load existing index
                self.index = faiss.read_index(str(index_file))
                
                # Check if the loaded index has the correct dimension
                if self.index.d != self.embedding_dimension:
                    print(f"⚠️ Existing index dimension ({self.index.d}) doesn't match expected ({self.embedding_dimension})")
                    print("Creating new index with correct dimension...")
                    self._create_new_index()
                    return
                
                with open(metadata_file, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                
                # Rebuild document mapping
                self._rebuild_document_mapping()
                print(f"✓ Loaded existing vector index with {len(self.chunk_metadata)} chunks")
                
            except Exception as e:
                print(f" Error loading existing index: {e}")
                print("Creating new index...")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _force_clean_start(self):
        """Force a completely clean start by removing all existing files and creating new index"""
        print(" Force clean start: Removing all existing index files")
        
        # Remove existing index files
        import os
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        if index_file.exists():
            os.remove(index_file)
            print("  ✓ Removed existing FAISS index file")
        if metadata_file.exists():
            os.remove(metadata_file)
            print("  ✓ Removed existing metadata file")
        
        # Create completely fresh index
        self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # IVF index with clustering
            quantizer = faiss.IndexFlatIP(self.embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, 100)
        elif self.index_type == "hnsw":
            # HNSW index for approximate nearest neighbor search
            self.index = faiss.IndexHNSWFlat(self.embedding_dimension, 32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        print(f"✓ Created new {self.index_type} FAISS index")
    
    def _rebuild_document_mapping(self):
        """Rebuild document mapping from chunk metadata"""
        self.document_mapping = {}
        for chunk_id, metadata in self.chunk_metadata.items():
            doc_id = metadata.get('document_id')
            if doc_id:
                if doc_id not in self.document_mapping:
                    self.document_mapping[doc_id] = []
                self.document_mapping[doc_id].append(chunk_id)
    
    def add_documents(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> bool:
        """
        Add document chunks and their embeddings to the vector store
        
        Args:
            chunks: List of TextChunk objects
            embeddings: List of embedding vectors (same order as chunks)
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks or not embeddings:
            print(" No chunks or embeddings provided")
            return False
        
        if len(chunks) != len(embeddings):
            print(" Number of chunks and embeddings must match")
            return False
        
        try:
            # Debug information
            print(f"   Debug: Expected embedding dimension: {self.embedding_dimension}")
            print(f"   Debug: First embedding length: {len(embeddings[0]) if embeddings else 'No embeddings'}")
            
            # Validate embedding dimensions
            for i, emb in enumerate(embeddings):
                if len(emb) != self.embedding_dimension:
                    raise ValueError(f"Embedding {i} has dimension {len(emb)}, expected {self.embedding_dimension}")
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Check array shape
            print(f"   Debug: Embeddings array shape: {embeddings_array.shape}")
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            if self.index_type == "ivf" and not self.index.is_trained:
                # Train IVF index if needed
                self.index.train(embeddings_array)
            
            self.index.add(embeddings_array)
            
            # Store chunk metadata using FAISS index position
            start_idx = self.index.ntotal - len(chunks)  # FAISS index position where chunks were added
            
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.chunk_id
                faiss_index = start_idx + i  # FAISS index position
                
                self.chunk_metadata[faiss_index] = {
                    'chunk_id': chunk_id,  # Store original chunk_id in metadata
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'text_length': len(chunk.text),
                    'word_count': chunk.metadata.get('word_count', 0),
                    'text_content': chunk.text,  # Store the actual text content
                    'metadata': chunk.metadata
                }
                
                # Update document mapping
                if chunk.document_id not in self.document_mapping:
                    self.document_mapping[chunk.document_id] = []
                self.document_mapping[chunk.document_id].append(faiss_index)
            
            print(f"✓ Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            import traceback
            print(f" Error adding documents to vector store: {e}")
            print(f" Error type: {type(e).__name__}")
            print(f" Full error details:")
            traceback.print_exc()
            return False
    
    def search(self, 
               query_embedding: List[float], 
               k: int = 5,
               document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using a query embedding
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            document_filter: Optional document ID to filter results
            
        Returns:
            List of search results with metadata
        """
        if self.index.ntotal == 0:
            print("Vector store is empty")
            return []
        
        try:
            print(f"    🔍 Vector store search: index has {self.index.ntotal} vectors, searching for {k} results")
            
            # Convert query to numpy array and normalize
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            # Perform search
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            print(f"     Search returned {len(scores[0])} results")
            print(f"     Raw scores: {scores[0]}")
            print(f"     Raw indices: {indices[0]}")
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                print(f"    Processing result {i}: score={score:.3f}, idx={idx}")
                
                if idx == -1:  # Invalid index
                    print(f"      Skipping invalid index {idx}")
                    continue
                
                # Get chunk metadata using FAISS index directly
                if idx not in self.chunk_metadata:
                    print(f" FAISS index {idx} not found in metadata")
                    continue
                    
                metadata = self.chunk_metadata[idx]
                chunk_id = metadata.get('chunk_id', f'chunk_{idx}')
                print(f"      📄 Found chunk: {chunk_id}")
                
                # Apply document filter if specified
                if document_filter and metadata.get('document_id') != document_filter:
                    print(f"   Filtered out by document filter")
                    continue
                
                result = {
                    'chunk_id': chunk_id,
                    'score': float(score),
                    'rank': i + 1,
                    'document_id': metadata.get('document_id'),
                    'chunk_index': metadata.get('chunk_index'),
                    'start_char': metadata.get('start_char'),
                    'end_char': metadata.get('end_char'),
                    'text_length': metadata.get('text_length'),
                    'word_count': metadata.get('word_count'),
                    'text_content': metadata.get('text_content', ''),  # Include actual text content
                    'metadata': metadata.get('metadata', {})
                }
                
                results.append(result)
                print(f"  Added result: {chunk_id} (score: {score:.3f})")
                
                if len(results) >= k:
                    break
            
            print(f"    📊 Final results: {len(results)} chunks")
            return results
            
        except Exception as e:
            print(f" Error searching vector store: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        if document_id not in self.document_mapping:
            return []
        
        chunks = []
        for chunk_id in self.document_mapping[document_id]:
            if chunk_id in self.chunk_metadata:
                metadata = self.chunk_metadata[chunk_id].copy()
                metadata['chunk_id'] = chunk_id
                chunks.append(metadata)
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        return chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_chunks': self.index.ntotal if self.index else 0,
            'total_documents': len(self.document_mapping),
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'index_size_mb': self._get_index_size_mb(),
            'documents': list(self.document_mapping.keys())
        }
    
    def _get_index_size_mb(self) -> float:
        """Get the size of the FAISS index in MB"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            if index_file.exists():
                return index_file.stat().st_size / (1024 * 1024)
        except:
            pass
        return 0.0
    
    def save_index(self) -> bool:
        """Save the FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_file = self.index_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.chunk_metadata, f)
            
            print(f" Saved vector index to {self.index_path}")
            return True
            
        except Exception as e:
            print(f" Error saving vector index: {e}")
            return False
    
    def clear_index(self) -> bool:
        """Clear all data from the vector store"""
        try:
            # Clear metadata first
            self.chunk_metadata.clear()
            self.document_mapping.clear()
            
            # Remove existing index files to force recreation
            import os
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "metadata.pkl"
            
            if index_file.exists():
                os.remove(index_file)
            if metadata_file.exists():
                os.remove(metadata_file)
            
            # Force complete recreation of index (this will clear all FAISS data)
            self.index = None
            self._create_new_index()
            
            print("✓ Completely cleared vector store and created fresh index")
            return True
            
        except Exception as e:
            print(f" Error clearing vector store: {e}")
            return False
    
    def remove_document(self, document_id: str) -> bool:
        """Remove all chunks for a specific document"""
        if document_id not in self.document_mapping:
            return False
        
        try:
            chunk_ids_to_remove = self.document_mapping[document_id].copy()
            
            # Remove from metadata
            for chunk_id in chunk_ids_to_remove:
                if chunk_id in self.chunk_metadata:
                    del self.chunk_metadata[chunk_id]
            
            # Remove from document mapping
            del self.document_mapping[document_id]
            
            # Note: FAISS doesn't support easy deletion, so we'll need to rebuild
            # For now, we'll just mark chunks as removed in metadata
            print(f" Document {document_id} marked for removal. Rebuild index to apply changes.")
            return True
            
        except Exception as e:
            print(f" Error removing document: {e}")
            return False
