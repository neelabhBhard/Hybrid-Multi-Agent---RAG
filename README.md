# Educational Content System - Hybrid Multi-Agent RAG

## Project Overview
This project implements a hybrid multi-agent system for educational content processing. The system combines deterministic agents (0 LLM calls) with intelligent LLM-based agents to efficiently process educational materials and generate quizzes.

## Architecture
- **Document Processor Agent**: Extracts text from PDFs (0 LLM calls)
- **Content Analyzer Agent**: Analyzes and understands content (1 LLM call)
- **Quiz Generator Agent**: Creates quizzes from analyzed content (0-1 LLM calls)

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the System
```bash
python main.py
```

## Features
- PDF document processing
- Content analysis and concept extraction
- Quiz generation with multiple question types
- LLM call tracking and cost estimation
- Efficient hybrid architecture (≤2 LLM calls per request)

## Project Structure
```
├── main.py                 # Main entry point
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   ├── document_processor.py
│   ├── content_analyzer.py
│   └── quiz_generator.py
├── core/                   # Core system components
│   ├── __init__.py
│   ├── llm_interface.py   # LLM management
│   ├── coordinator.py     # Agent coordination
│   └── models.py          # Data models
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── file_utils.py      # File handling utilities
└── requirements.txt        # Python dependencies
```

## Usage Examples
1. **Simple Quiz Generation**: Upload a lecture PDF and generate practice questions
2. **Content Analysis**: Extract key concepts and topics from educational materials
3. **Custom Quiz Types**: Generate different question formats (multiple choice, short answer)

## Cost Efficiency
- Target: ≤2 LLM calls per interaction
- Estimated cost: $0.002 per GPT-3.5-turbo call
- System tracks and reports all LLM usage