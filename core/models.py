"""
Data Models for the Educational Content System
Uses Pydantic for validation and structure
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QuestionType(str, Enum):
    """Types of questions that can be generated"""
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"
    FILL_IN_BLANK = "fill_in_blank"


class Question(BaseModel):
    """Represents a single quiz question"""
    question_text: str = Field(..., description="The question text")
    question_type: QuestionType = Field(..., description="Type of question")
    correct_answer: str = Field(..., description="Correct answer")
    options: Optional[List[str]] = Field(None, description="Multiple choice options")
    explanation: Optional[str] = Field(None, description="Explanation of the answer")
    difficulty: str = Field("medium", description="Question difficulty level")
    topic: Optional[str] = Field(None, description="Topic this question covers")


class Quiz(BaseModel):
    """Represents a complete quiz"""
    title: str = Field(..., description="Quiz title")
    description: str = Field(..., description="Quiz description")
    questions: List[Question] = Field(..., description="List of questions")
    total_questions: int = Field(..., description="Total number of questions")
    estimated_time: int = Field(..., description="Estimated time in minutes")
    difficulty: str = Field("medium", description="Overall quiz difficulty")


class ContentConcept(BaseModel):
    """Represents a key concept extracted from content"""
    concept_name: str = Field(..., description="Name of the concept")
    description: str = Field(..., description="Description of the concept")
    importance_score: float = Field(..., ge=0, le=1, description="Importance score 0-1")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    examples: List[str] = Field(default_factory=list, description="Examples of the concept")


class ContentAnalysis(BaseModel):
    """Represents the analysis of educational content"""
    main_topics: List[str] = Field(..., description="Main topics covered")
    key_concepts: List[ContentConcept] = Field(..., description="Key concepts identified")
    difficulty_level: str = Field(..., description="Overall difficulty level")
    estimated_reading_time: int = Field(..., description="Estimated reading time in minutes")
    summary: str = Field(..., description="Brief summary of content")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")


class DocumentInfo(BaseModel):
    """Information about the processed document"""
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    pages: int = Field(..., description="Number of pages")
    text_length: int = Field(..., description="Length of extracted text")
    processing_time: float = Field(..., description="Time taken to process in seconds")


class ProcessingResult(BaseModel):
    """Result of processing a document through the system"""
    document_info: DocumentInfo = Field(..., description="Document information")
    content_analysis: Optional[ContentAnalysis] = Field(None, description="Content analysis")
    quiz: Optional[Quiz] = Field(None, description="Generated quiz")
    llm_calls_used: int = Field(..., description="Number of LLM calls used")
    total_cost: float = Field(..., description="Total cost of processing")
    processing_time: float = Field(..., description="Total processing time")
    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class SystemStats(BaseModel):
    """System performance statistics"""
    total_documents_processed: int = Field(0, description="Total documents processed")
    successful_documents: int = Field(0, description="Total successfully processed documents")
    total_llm_calls: int = Field(0, description="Total LLM calls made")
    total_cost: float = Field(0.0, description="Total cost incurred")
    average_llm_calls_per_document: float = Field(0.0, description="Average LLM calls per document")
    success_rate: float = Field(0.0, description="Success rate as percentage")
