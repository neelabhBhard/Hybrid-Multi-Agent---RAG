"""
Main Entry Point for the Educational Content System
Demonstrates the hybrid multi-agent architecture
"""

import os
import sys
from core.coordinator import Coordinator
import time


def main():
    """Main function to run the Educational Content System"""
    print("EDUCATIONAL CONTENT SYSTEM")
    print("=" * 50)
    print("Hybrid Multi-Agent System with RAG")
    print("=" * 50)
    
    try:
        # Check for API key (OpenAI or Anthropic)
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: No API key found")
            print("Please create a .env file with either:")
            print("ANTHROPIC_API_KEY=your_anthropic_key_here")
            print("OPENAI_API_KEY=your_openai_key_here")
            return
        
        # Initialize the system
        print("\nInitializing Educational Content System...")
        coordinator = Coordinator(api_key)
        
        # Main menu
        while True:
            print("\n" + "=" * 50)
            print("MAIN MENU")
            print("=" * 50)
            print("1. Process a PDF document")
            print("2. Index document for RAG")
            print("3. Ask RAG query")
            print("4. Use calculator tool")
            print("5. View system statistics")
            print("6. Reset system statistics")
            print("7. View embedding service info")
            print("8. Exit")
            print("=" * 50)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                process_document_menu(coordinator)
            elif choice == "2":
                index_document_menu(coordinator)
            elif choice == "3":
                rag_query_menu(coordinator)
            elif choice == "4":
                calculator_menu(coordinator)
            elif choice == "5":
                coordinator.display_system_stats()
            elif choice == "6":
                coordinator.reset_system_stats()
                print("System statistics reset")
            elif choice == "7":
                embedding_info = coordinator.get_embedding_service_info()
                print("\nEMBEDDING SERVICE INFORMATION")
                print("=" * 50)
                print(f"Model: {embedding_info['model_name']}")
                print(f"Dimension: {embedding_info['embedding_dimension']}")
                print(f"Status: {'Loaded' if embedding_info['model_loaded'] else 'Fallback Mode'}")
                if embedding_info['fallback_mode']:
                    print("Note: Using fallback embeddings. Install sentence-transformers for better results.")
            elif choice == "8":
                print("Goodbye! Thank you for using the Educational Content System.")
                break
            else:
                print("Invalid choice. Please enter 1-8.")
    
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        print("Please check your configuration and try again.")


def process_document_menu(coordinator: Coordinator):
    """Handle document processing menu"""
    print("\nDOCUMENT PROCESSING")
    print("-" * 30)
    
    # Get file path
    file_path = input("Enter the path to your PDF file: ").strip()
    
    if not file_path:
        print("No file path provided")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    if not file_path.lower().endswith(('.pdf', '.txt')):
        print("Only PDF and TXT files are supported")
        return
    
    # Quiz generation options
    print("\nQuiz Generation Options:")
    generate_quiz = input("Generate quiz? (y/n, default: y): ").strip().lower()
    generate_quiz = generate_quiz != 'n'
    
    num_questions = 5
    if generate_quiz:
        try:
            num_input = input("Number of questions (default: 5): ").strip()
            if num_input:
                num_questions = int(num_input)
                if num_questions < 1 or num_questions > 20:
                    print("Number of questions limited to 1-20. Using default: 5")
                    num_questions = 5
        except ValueError:
            print("Invalid number. Using default: 5 questions")
    
    # Process the document
    try:
        print(f"\nProcessing document: {os.path.basename(file_path)}")
        result = coordinator.process_document(
            file_path=file_path,
            generate_quiz=generate_quiz,
            num_questions=num_questions
        )
        
        if result.success:
            # Display quiz if generated
            if result.quiz:
                display_quiz(result.quiz)
            
            # Ask if user wants to save results
            save_results = input("\nSave results to file? (y/n): ").strip().lower()
            if save_results == 'y':
                save_results_to_file(result, file_path)
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")


def display_quiz(quiz):
    """Display the generated quiz in a readable format"""
    print(f"\nQUIZ: {quiz.title}")
    print(f"Description: {quiz.description}")
    print(f"Estimated Time: {quiz.estimated_time} minutes")
    print(f"Difficulty: {quiz.difficulty}")
    print(f"Questions: {quiz.total_questions}")
    print("-" * 60)
    
    for i, question in enumerate(quiz.questions, 1):
        print(f"\nQ{i}. {question.question_text}")
        print(f"   Type: {question.question_type.value}")
        
        if question.options:
            for j, option in enumerate(question.options, 1):
                print(f"   {j}. {option}")
        
        print(f"   Answer: {question.correct_answer}")
        if question.explanation:
            print(f"   {question.explanation}")
        print(f"   Topic: {question.topic}")
        print(f"   Difficulty: {question.difficulty}")


def save_results_to_file(result, original_file_path):
    """Save processing results to a text file"""
    try:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        output_file = f"{base_name}_results.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("EDUCATIONAL CONTENT SYSTEM - PROCESSING RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Document info
            f.write("DOCUMENT INFORMATION:\n")
            f.write(f"Filename: {result.document_info.filename}\n")
            f.write(f"Pages: {result.document_info.pages}\n")
            f.write(f"Text Length: {result.document_info.text_length:,} characters\n")
            f.write(f"Processing Time: {result.document_info.processing_time:.2f}s\n\n")
            
            # Content analysis
            if result.content_analysis:
                f.write("CONTENT ANALYSIS:\n")
                f.write(f"Main Topics: {', '.join(result.content_analysis.main_topics)}\n")
                f.write(f"Difficulty Level: {result.content_analysis.difficulty_level}\n")
                f.write(f"Estimated Reading Time: {result.content_analysis.estimated_reading_time} minutes\n")
                f.write(f"Summary: {result.content_analysis.summary}\n\n")
                
                f.write("Key Concepts:\n")
                for concept in result.content_analysis.key_concepts:
                    f.write(f"• {concept.concept_name}: {concept.description}\n")
                    f.write(f"  Importance: {concept.importance_score:.2f}\n")
                    if concept.examples:
                        f.write(f"  Examples: {', '.join(concept.examples)}\n")
                    f.write("\n")
            
            # Quiz
            if result.quiz:
                f.write("GENERATED QUIZ:\n")
                f.write(f"Title: {result.quiz.title}\n")
                f.write(f"Description: {result.quiz.description}\n")
                f.write(f"Questions: {result.quiz.total_questions}\n")
                f.write(f"Difficulty: {result.quiz.difficulty}\n")
                f.write(f"Estimated Time: {result.quiz.estimated_time} minutes\n\n")
                
                for i, question in enumerate(result.quiz.questions, 1):
                    f.write(f"Q{i}. {question.question_text}\n")
                    f.write(f"   Type: {question.question_type.value}\n")
                    if question.options:
                        for j, option in enumerate(question.options, 1):
                            f.write(f"   {j}. {option}\n")
                    f.write(f"   Answer: {question.correct_answer}\n")
                    if question.explanation:
                        f.write(f"   Explanation: {question.explanation}\n")
                    f.write(f"   Topic: {question.topic}\n")
                    f.write(f"   Difficulty: {question.difficulty}\n\n")
            
            # Processing stats
            f.write("PROCESSING STATISTICS:\n")
            f.write(f"LLM Calls Used: {result.llm_calls_used}\n")
            f.write(f"Total Cost: ${result.total_cost:.4f}\n")
            f.write(f"Total Processing Time: {result.processing_time:.2f}s\n")
            f.write(f"Success: {result.success}\n")
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")


def index_document_menu(coordinator: Coordinator):
    """Handle document indexing for RAG menu"""
    print("\nDOCUMENT INDEXING FOR RAG")
    print("-" * 30)
    
    # Get file path
    file_path = input("Enter the path to your document file: ").strip()
    
    if not file_path:
        print("No file path provided")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    # Check file type
    if not file_path.lower().endswith(('.pdf', '.txt')):
        print("Only PDF and TXT files are supported for indexing")
        return
    
    print(f"\nIndexing document: {file_path}")
    print("This will extract text, create chunks, and store them in the vector database.")
    
    # Confirm indexing
    confirm = input("Proceed with indexing? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Indexing cancelled")
        return
    
    try:
        # Index the document
        result = coordinator.index_document_for_rag(file_path)
        
        if result['success']:
            print(f"\nDocument indexed successfully!")
            print(f"Chunks created: {result['chunks_created']}")
            
            # Show chunking statistics
            if 'chunking_stats' in result:
                stats = result['chunking_stats']
                print(f"Chunking Statistics:")
                print(f"  • Average chunk size: {stats.get('average_chunk_size', 0):.0f} characters")
                print(f"  • Min chunk size: {stats.get('min_chunk_size', 0)} characters")
                print(f"  • Max chunk size: {stats.get('max_chunk_size', 0)} characters")
            
            # Show vector store statistics
            if 'vector_store_stats' in result:
                stats = result['vector_store_stats']
                print(f" Vector Store Statistics:")
                print(f"  • Total chunks: {stats.get('total_chunks', 0)}")
                print(f"  • Total documents: {stats.get('total_documents', 0)}")
                print(f"  • Index size: {stats.get('index_size_mb', 0):.2f} MB")
        else:
            print(f"\nIndexing failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f" Error during indexing: {str(e)}")


def rag_query_menu(coordinator: Coordinator):
    """Handle RAG query menu"""
    print("\n RAG QUERY SYSTEM")
    print("-" * 30)
    
    # Check if vector store has documents
    vector_stats = coordinator.get_vector_store_stats()
    if vector_stats['total_chunks'] == 0:
        print(" No documents indexed yet!")
        print("Please index some documents first using option 2.")
        return
    
    print(f" Available documents: {vector_stats['total_documents']}")
    print(f" Total chunks: {vector_stats['total_chunks']}")
    
    # Get query
    query = input("\nEnter your question: ").strip()
    
    if not query:
        print(" No query provided")
        return
    
    # Optional document filter
    if vector_stats['total_documents'] > 1:
        print(f"\n Available documents:")
        for i, doc_id in enumerate(vector_stats['documents'], 1):
            print(f"  {i}. {doc_id}")
        
        filter_choice = input("\nFilter by specific document? (enter number or press Enter for all): ").strip()
        document_filter = None
        if filter_choice:
            try:
                doc_index = int(filter_choice) - 1
                if 0 <= doc_index < len(vector_stats['documents']):
                    document_filter = vector_stats['documents'][doc_index]
                    print(f"✓ Filtering by: {document_filter}")
            except ValueError:
                print(" Invalid choice, searching all documents")
    else:
        document_filter = None
    
    try:
        # Process RAG query
        result = coordinator.process_rag_query(query, document_filter)
        
        if result['success']:
            # Ask if user wants to save the response
            save_choice = input("\nSave response to file? (y/n): ").strip().lower()
            if save_choice == 'y':
                save_rag_response_to_file(result, query)
        else:
            print(f"\n Query failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"Error processing query: {str(e)}")


def calculator_menu(coordinator: Coordinator):
    """Handle calculator tool menu"""
    print("\nCALCULATOR TOOL")
    print("-" * 30)
    
    while True:
        expression = input("Enter mathematical expression (or 'back' to return): ").strip()
        
        if expression.lower() == 'back':
            break
        
        if expression:
            coordinator.calculate_expression(expression)
        else:
            print("No expression provided")


def save_rag_response_to_file(result: dict, query: str):
    """Save RAG response to a text file"""
    try:
        # Create output filename
        output_file = f"rag_response_{int(time.time())}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("RAG QUERY RESPONSE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Query: {query}\n")
            f.write(f"Response: {result['response']}\n\n")
            
            f.write("Retrieved Chunks:\n")
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                f.write(f"Chunk {i}:\n")
                f.write(f"  Document: {chunk.get('document_id', 'unknown')}\n")
                f.write(f"  Score: {chunk.get('score', 0):.3f}\n")
                f.write(f"  Position: {chunk.get('start_char', 0)}-{chunk.get('end_char', 0)}\n\n")
            
            f.write("Processing Information:\n")
            f.write(f"  Processing Time: {result.get('processing_time', 0):.2f}s\n")
            f.write(f"  LLM Calls Used: {result.get('llm_calls_used', 0)}\n")
            f.write(f"  Retrieved Chunks: {len(result.get('retrieved_chunks', []))}\n")
        
        print(f" RAG response saved to: {output_file}")
        
    except Exception as e:
        print(f" Error saving RAG response: {str(e)}")


if __name__ == "__main__":
    main()
