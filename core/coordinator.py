"""
Coordinator for the Educational Content System
Manages all agents and uses deterministic routing (no LLM calls)
"""

import time
from typing import Optional, Dict, Any
from agents.document_processor import DocumentProcessorAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.quiz_generator import QuizGeneratorAgent
from agents.rag_agent import RAGAgent
from core.llm_interface import LLMInterface
from core.vector_store import VectorStore
from core.embedding_service import EmbeddingService
from core.models import ProcessingResult, DocumentInfo, ContentAnalysis, Quiz, SystemStats
from utils.calculator_tool import CalculatorTool


class Coordinator:
    """Coordinates all agents in the educational content system"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Initialize LLM interface
        self.llm_interface = LLMInterface(api_key)
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        
        # Initialize vector store for RAG with embedding service dimension
        self.vector_store = VectorStore(embedding_dimension=self.embedding_service.embedding_dimension, force_clean_start=True)
        
        # Initialize all agents
        self.document_processor = DocumentProcessorAgent()
        self.content_analyzer = ContentAnalyzerAgent(self.llm_interface)
        self.quiz_generator = QuizGeneratorAgent(self.llm_interface)
        self.rag_agent = RAGAgent(self.llm_interface, self.vector_store, self.embedding_service)
        
        # Initialize additional tools
        self.calculator_tool = CalculatorTool()
        
        # System statistics
        self.system_stats = SystemStats()
        
        print(" Educational Content System initialized!")
        print(f"  Document Processor: {self.document_processor}")
        print(f" Content Analyzer: {self.content_analyzer}")
        print(f" Quiz Generator: {self.quiz_generator}")
        print(f"  RAG Agent: {self.rag_agent}")
        print(f" Calculator Tool: {self.calculator_tool}")
    
    def process_document(self, file_path: str, generate_quiz: bool = True, num_questions: int = 5) -> ProcessingResult:
        """
        Process a document through the entire system
        
        Args:
            file_path: Path to the PDF file
            generate_quiz: Whether to generate a quiz
            num_questions: Number of quiz questions to generate
            
        Returns:
            ProcessingResult with all processing information
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f" PROCESSING DOCUMENT: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Document Processing (0 LLM calls)
            print(f"\n STEP 1: Document Processing")
            document_info = self.document_processor.process(file_path)
            extracted_text = self.document_processor.get_extracted_text()
            
            if not extracted_text:
                raise ValueError("No text was extracted from the document")
            
            # Step 2: Content Analysis (1 LLM call)
            print(f"\n STEP 2: Content Analysis")
            content_analysis = self.content_analyzer.process(extracted_text)
            
            # Step 3: Quiz Generation (0-1 LLM calls)
            quiz = None
            if generate_quiz:
                print(f"\nSTEP 3: Quiz Generation")
                quiz = self.quiz_generator.process(content_analysis, num_questions)
            
            # Calculate total processing time
            total_processing_time = time.time() - start_time
            
            # Get LLM usage statistics
            llm_stats = self.llm_interface.get_usage_stats()
            
            # Create processing result
            result = ProcessingResult(
                document_info=document_info,
                content_analysis=content_analysis,
                quiz=quiz,
                llm_calls_used=llm_stats['total_calls'],
                total_cost=llm_stats['total_cost'],
                processing_time=total_processing_time,
                success=True
            )
            
            # Update system statistics
            self._update_system_stats(result)
            
            # Display results
            self._display_processing_summary(result)
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            processing_time = time.time() - start_time
            llm_stats = self.llm_interface.get_usage_stats()
            
            error_result = ProcessingResult(
                document_info=DocumentInfo(
                    filename=file_path.split('/')[-1] if '/' in file_path else file_path,
                    file_size=0,
                    pages=0,
                    text_length=0,
                    processing_time=processing_time
                ),
                llm_calls_used=llm_stats['total_calls'],
                total_cost=llm_stats['total_cost'],
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
            
            print(f"\n Processing failed: {str(e)}")
            return error_result
    
    def _update_system_stats(self, result: ProcessingResult):
        """Update system-wide statistics"""
        self.system_stats.total_documents_processed += 1
        self.system_stats.total_llm_calls += result.llm_calls_used
        self.system_stats.total_cost += result.total_cost
        
        # Track successful vs failed documents
        if result.success:
            self.system_stats.successful_documents += 1
        
        # Calculate success rate as percentage
        self.system_stats.success_rate = (
            self.system_stats.successful_documents / 
            self.system_stats.total_documents_processed
        ) * 100
        
        # Calculate average LLM calls per document
        self.system_stats.average_llm_calls_per_document = (
            self.system_stats.total_llm_calls / 
            self.system_stats.total_documents_processed
        )
    
    def _display_processing_summary(self, result: ProcessingResult):
        """Display a summary of the processing results"""
        print(f"\n{'='*60}")
        print(f" PROCESSING COMPLETE!")
        print(f"{'='*60}")
        
        print(f" Document Information:")
        print(f"  • Filename: {result.document_info.filename}")
        print(f"  • Pages: {result.document_info.pages}")
        print(f"  • Text Length: {result.document_info.text_length:,} characters")
        
        if result.content_analysis:
            print(f"\n Content Analysis:")
            print(f"  • Main Topics: {len(result.content_analysis.main_topics)}")
            print(f"  • Key Concepts: {len(result.content_analysis.key_concepts)}")
            print(f"  • Difficulty: {result.content_analysis.difficulty_level}")
            print(f"  • Reading Time: ~{result.content_analysis.estimated_reading_time} minutes")
        
        if result.quiz:
            print(f"\n Quiz Generated:")
            print(f"  • Title: {result.quiz.title}")
            print(f"  • Questions: {result.quiz.total_questions}")
            print(f"  • Difficulty: {result.quiz.difficulty}")
            print(f"  • Estimated Time: {result.quiz.estimated_time} minutes")
        
        print(f"\n Cost Analysis:")
        print(f"  • LLM Calls Used: {result.llm_calls_used}")
        print(f"  • Total Cost: ${result.total_cost:.4f}")
        print(f"  • Processing Time: {result.processing_time:.2f}s")
        
        # Efficiency check
        if result.llm_calls_used <= 2:
            print(f"  Efficiency: Excellent (≤2 LLM calls)")
        elif result.llm_calls_used <= 3:
            print(f"  Efficiency: Good (≤3 LLM calls)")
        else:
            print(f"  Efficiency: Needs improvement (>3 LLM calls)")
    
    def get_system_stats(self) -> SystemStats:
        """Get current system statistics"""
        return self.system_stats
    
    def display_system_stats(self):
        """Display comprehensive system statistics"""
        print(f"\n{'='*60}")
        print(f"SYSTEM STATISTICS")
        print(f"{'='*60}")
        
        stats = self.system_stats
        print(f" Documents Processed: {stats.total_documents_processed}")
        print(f"Successful Documents: {stats.successful_documents}")
        print(f" Failed Documents: {stats.total_documents_processed - stats.successful_documents}")
        print(f" Total LLM Calls: {stats.total_llm_calls}")
        print(f" Total Cost: ${stats.total_cost:.4f}")
        print(f" Avg LLM Calls/Doc: {stats.average_llm_calls_per_document:.2f}")
        print(f" Success Rate: {stats.success_rate:.1f}%")
        
        # Agent-specific statistics
        print(f"\n AGENT STATISTICS:")
        print(f"   {self.document_processor.get_stats()}")
        print(f"   {self.content_analyzer.get_stats()}")
        print(f"   {self.quiz_generator.get_stats()}")
        print(f"   {self.rag_agent.get_stats()}")
        
        # LLM interface statistics
        llm_stats = self.llm_interface.get_usage_stats()
        print(f"\n LLM INTERFACE:")
        print(f"  • Total Calls: {llm_stats['total_calls']}")
        print(f"  • Total Tokens: {llm_stats['total_tokens']:,}")
        print(f"  • Average Cost/Call: ${llm_stats['average_cost_per_call']:.4f}")
    
    def reset_system_stats(self):
        """Reset all system statistics (useful for testing)"""
        self.system_stats = SystemStats()
        self.llm_interface.reset_stats()
        print("✓ System statistics reset")
    
    def get_agent_stats(self) -> dict:
        """Get statistics from all agents"""
        return {
            "document_processor": self.document_processor.get_stats(),
            "content_analyzer": self.content_analyzer.get_stats(),
            "quiz_generator": self.quiz_generator.get_stats(),
            "rag_agent": self.rag_agent.get_stats(),
            "llm_interface": self.llm_interface.get_usage_stats()
        }
    
    def process_rag_query(self, query: str, document_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a RAG query using the RAG agent
        
        Args:
            query: User query string
            document_filter: Optional document ID to filter results
            
        Returns:
            Dictionary with RAG response and metadata
        """
        print(f"\n{'='*60}")
        print(f"🔍 RAG QUERY: {query}")
        print(f"{'='*60}")
        
        try:
            # Process query through RAG agent
            result = self.rag_agent.process({
                'query': query,
                'document_filter': document_filter
            })
            
            if result['success']:
                print(f"\n RAG Query Successful!")
                print(f" Response: {result['response']}")
                print(f" Retrieved Chunks: {len(result['retrieved_chunks'])}")
                print(f"  Processing Time: {result['processing_time']:.2f}s")
                print(f" LLM Calls Used: {result['llm_calls_used']}")
            else:
                print(f"\n RAG Query Failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"\n Error processing RAG query: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response': None,
                'retrieved_chunks': [],
                'llm_calls_used': 0
            }
    
    def calculate_expression(self, expression: str) -> Dict[str, Any]:
        """
        Calculate a mathematical expression using the calculator tool
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Dictionary with calculation result and metadata
        """
        print(f"\n{'='*60}")
        print(f"CALCULATOR: {expression}")
        print(f"{'='*60}")
        
        try:
            # Use calculator tool
            result = self.calculator_tool.calculate(expression)
            
            if result.success:
                print(f"\n Calculation Successful!")
                print(f" Result: {result.result}")
            else:
                print(f"\n Calculation Failed: {result.error_message}")
            
            return {
                'success': result.success,
                'result': result.result,
                'error_message': result.error_message
            }
            
        except Exception as e:
            print(f"\n Error in calculation: {str(e)}")
            return {
                'success': False,
                'result': None,
                'error_message': str(e)
            }
    
    def solve_equation(self, equation: str) -> Dict[str, Any]:
        """
        Solve a mathematical equation using the calculator tool
        
        Args:
            equation: Mathematical equation string
            
        Returns:
            Dictionary with solution and metadata
        """
        print(f"\n{'='*60}")
        print(f"EQUATION SOLVER: {equation}")
        print(f"{'='*60}")
        
        try:
            # Use calculator tool's equation solver
            result = self.calculator_tool.solve_equation(equation)
            
            if result.success:
                print(f"\n Equation Solved!")
                print(f"Solution: {result.result}")
            else:
                print(f"\n Equation Solving Failed: {result.error_message}")
            
            return {
                'success': result.success,
                'result': result.result,
                'error_message': result.error_message
            }
            
        except Exception as e:
            print(f"\n Error solving equation: {str(e)}")
            return {
                'success': False,
                'result': None,
                'error_message': str(e)
            }
    
    def index_document_for_rag(self, file_path: str) -> Dict[str, Any]:
        """
        Index a document for RAG retrieval
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with indexing results
        """
        print(f"\n{'='*60}")
        print(f" INDEXING DOCUMENT FOR RAG: {file_path}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Extract text from document
            print(f"\nSTEP 1: Text Extraction")
            document_info = self.document_processor.process(file_path)
            extracted_text = self.document_processor.get_extracted_text()
            
            if not extracted_text:
                raise ValueError("No text was extracted from the document")
            
            # Step 2: Chunk the text
            print(f"\n STEP 2: Text Chunking")
            from utils.text_chunker import TextChunker
            chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
            chunks = chunker.chunk_text(extracted_text, document_info.filename)
            
            if not chunks:
                raise ValueError("No chunks were created from the document")
            
            print(f"  ✓ Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings using the embedding service
            print(f"\n STEP 3: Embedding Generation")
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            print(f"  ✓ Generated {len(embeddings)} embeddings")
            
            # Step 4: Add to vector store
            print(f"\n STEP 4: Vector Store Indexing")
            success = self.vector_store.add_documents(chunks, embeddings)
            
            if success:
                # Save the index
                self.vector_store.save_index()
                print(f"  ✓ Document indexed successfully")
                
                # Get indexing statistics
                chunk_stats = chunker.get_chunking_stats(chunks)
                vector_stats = self.vector_store.get_statistics()
                
                return {
                    'success': True,
                    'document_info': document_info,
                    'chunks_created': len(chunks),
                    'chunking_stats': chunk_stats,
                    'vector_store_stats': vector_stats,
                    'llm_calls_used': 0  # No LLM calls for indexing
                }
            else:
                raise ValueError("Failed to add documents to vector store")
                
        except Exception as e:
            print(f"\n Error indexing document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'chunks_created': 0,
                'llm_calls_used': 0
            }
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.vector_store.get_statistics()
    
    def get_calculator_capabilities(self) -> Dict[str, Any]:
        """Get calculator tool capabilities"""
        operations = self.calculator_tool.get_supported_operations()
        return {
            'supported_operations': {
                'basic': operations['basic'],
                'advanced': operations['advanced'],
                'constants': operations['constants']
            },
            'math_constants': self.calculator_tool.math_constants
        }
    
    def get_embedding_service_info(self) -> Dict[str, Any]:
        """Get embedding service information"""
        return self.embedding_service.get_model_info()
