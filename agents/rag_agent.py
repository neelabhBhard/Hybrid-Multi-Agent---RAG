"""
RAG (Retrieval-Augmented Generation) Agent
Handles knowledge retrieval and context-aware response generation
"""

import time
from typing import List, Dict, Any, Optional
from agents.base_agent import BaseAgent
from core.llm_interface import LLMInterface
from core.vector_store import VectorStore
from core.models import ContentAnalysis
from core.embedding_service import EmbeddingService


class RAGAgent(BaseAgent):
    """Agent for retrieval-augmented generation"""
    
    def __init__(self, 
                 llm_interface: LLMInterface,
                 vector_store: VectorStore,
                 embedding_service: EmbeddingService,
                 name: str = "RAG Agent"):
        """
        Initialize the RAG agent
        
        Args:
            llm_interface: Interface for LLM calls
            vector_store: Vector store for document retrieval
            embedding_service: Service for generating embeddings
            name: Agent name
        """
        super().__init__(name=name, uses_llm=True)
        self.llm_interface = llm_interface
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_context_chunks = 5
        self.min_similarity_score = 0.1  # Lower threshold for better retrieval
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process a query using RAG
        
        Args:
            input_data: Query string or dict with query and context
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Extract query and context
            if isinstance(input_data, str):
                query = input_data
                document_filter = None
            elif isinstance(input_data, dict):
                query = input_data.get('query', '')
                document_filter = input_data.get('document_filter')
            else:
                raise ValueError("Input must be string or dict with 'query' key")
            
            if not query.strip():
                return {
                    'success': False,
                    'error': 'Empty query provided',
                    'response': None,
                    'retrieved_chunks': [],
                    'llm_calls_made': 0
                }
            
            print(f" {self.name} processing query: '{query[:50]}...'")
            
            # Step 1: Generate query embedding using the embedding service
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(
                query_embedding, 
                document_filter
            )
            
            if not retrieved_chunks:
                return {
                    'success': False,
                    'error': 'No relevant information found',
                    'response': None,
                    'retrieved_chunks': [],
                    'llm_calls_made': 0
                }
            
            # Step 3: Generate response using LLM
            response = self._generate_rag_response(query, retrieved_chunks)
            
            # Track LLM call
            self.track_llm_call()
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            return {
                'success': True,
                'response': response,
                'retrieved_chunks': retrieved_chunks,
                'query': query,
                'processing_time': processing_time,
                'llm_calls_made': 1,
                'llm_calls_used': 1,  # Add missing key
                'similarity_scores': [chunk['score'] for chunk in retrieved_chunks]
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'retrieved_chunks': [],
                'processing_time': processing_time,
                'llm_calls_made': 0,
                'llm_calls_used': 0  # Add missing key
            }
    

    
    def _retrieve_relevant_chunks(self, 
                                 query_embedding: List[float], 
                                 document_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from the vector store
        
        Args:
            query_embedding: Query vector
            document_filter: Optional document ID filter
            
        Returns:
            List of relevant chunks with metadata
        """
        print(f" Retrieving relevant chunks...")
        
        # Search for similar chunks
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            k=self.max_context_chunks * 2,  # Get more results for filtering
            document_filter=document_filter
        )
        
        print(f" Raw search results: {len(search_results)} chunks found")
        if search_results:
            print(f" Score range: {min(r['score'] for r in search_results):.3f} to {max(r['score'] for r in search_results):.3f}")
        
        # Filter by similarity score and limit results
        relevant_chunks = []
        for result in search_results:
            if result['score'] >= self.min_similarity_score:
                relevant_chunks.append(result)
                if len(relevant_chunks) >= self.max_context_chunks:
                    break
        
        print(f"  Retrieved {len(relevant_chunks)} relevant chunks (threshold: {self.min_similarity_score})")
        return relevant_chunks
    
    def _generate_rag_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the retrieved chunks and LLM
        
        Args:
            query: User query
            chunks: Retrieved relevant chunks
            
        Returns:
            Generated response
        """
        print(f" Generating RAG response...")
        
        # Prepare context from chunks
        context = self._prepare_context(chunks)
        
        # Create prompt for LLM
        prompt = self._create_rag_prompt(query, context)
        
        # Generate response using LLM
        try:
            response = self.llm_interface.make_call(
                messages=prompt,
                model="gpt-3.5-turbo" if "openai" in str(type(self.llm_interface.client)) else "claude-3-5-sonnet-20240620",
                temperature=0.3,
                max_tokens=500
            )
            
            return response
            
        except Exception as e:
            print(f" LLM call failed: {e}")
            # Fallback response
            return self._create_fallback_response(query, chunks)
    
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from retrieved chunks
        
        Args:
            chunks: List of chunk metadata
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            # Get the actual chunk text from the vector store
            chunk_text = self._get_chunk_text(chunk)
            
            if chunk_text:
                context_parts.append(f"Chunk {i+1} (Score: {chunk['score']:.3f}):\n{chunk_text}\n")
        
        return "\n".join(context_parts)
    
    def _get_chunk_text(self, chunk: Dict[str, Any]) -> str:
        """
        Get the actual text content of a chunk
        
        Args:
            chunk: Chunk metadata from vector store
            
        Returns:
            Chunk text content
        """
        # Get the actual text content from the chunk metadata
        text_content = chunk.get('text_content', '')
        
        if text_content:
            return text_content
        
        # Fallback: if no text_content, try to get from metadata
        document_id = chunk.get('document_id', 'unknown')
        chunk_index = chunk.get('chunk_index', 0)
        
        return f"[Content from {document_id}, chunk {chunk_index} - No text content available]"
    
    def _create_rag_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        Create the prompt for RAG response generation
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            List of message dictionaries for LLM
        """
        system_message = """You are an educational assistant. Use the provided context to answer questions accurately and comprehensively. If the context doesn't contain enough information to answer the question, say so clearly. Always cite which chunks of information you used in your response."""

        user_message = f"""Question: {query}

Context Information:
{context}

Please provide a comprehensive answer based on the context above. If you use information from specific chunks, mention which ones. If the context doesn't contain enough information, please state that clearly."""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _create_fallback_response(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Create a fallback response when LLM fails
        
        Args:
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Fallback response
        """
        chunk_info = []
        for chunk in chunks:
            doc_id = chunk.get('document_id', 'unknown')
            score = chunk.get('score', 0.0)
            chunk_info.append(f"- {doc_id} (relevance: {score:.3f})")
        
        return f"""Relevant information found for query: "{query}"

Available sources:
{chr(10).join(chunk_info)}

Note: A detailed response could not be generated due to a technical issue. The retrieved chunks contain relevant information that could help answer your question."""

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        vector_stats = self.vector_store.get_statistics()
        
        return {
            'agent_name': self.name,
            'llm_calls_made': self.llm_calls_made,
            'total_processing_time': self.total_processing_time,
            'vector_store_stats': vector_stats,
            'max_context_chunks': self.max_context_chunks,
            'min_similarity_score': self.min_similarity_score
        }
    
    def update_retrieval_parameters(self, 
                                   max_context_chunks: Optional[int] = None,
                                   min_similarity_score: Optional[float] = None):
        """
        Update retrieval parameters
        
        Args:
            max_context_chunks: Maximum number of context chunks to use
            min_similarity_score: Minimum similarity score for chunks
        """
        if max_context_chunks is not None:
            self.max_context_chunks = max_context_chunks
            print(f"  ✓ Updated max context chunks to {max_context_chunks}")
        
        if min_similarity_score is not None:
            self.min_similarity_score = min_similarity_score
            print(f"  ✓ Updated min similarity score to {min_similarity_score}")
