"""
Quiz Generator Agent
Generates quizzes from analyzed content
"""

import time
from typing import List, Optional
from agents.base_agent import BaseAgent
from core.models import Quiz, Question, QuestionType, ContentAnalysis
from core.llm_interface import LLMInterface


class QuizGeneratorAgent(BaseAgent):
    """Agent for generating quizzes from analyzed content"""
    
    def __init__(self, llm_interface: LLMInterface):
        super().__init__(name="Quiz Generator", uses_llm=False)
        self.llm = llm_interface  # Keep for potential future use but don't track calls
    
    def process(self, content_analysis: ContentAnalysis, num_questions: int = 5) -> Quiz:
        """
        Generate a quiz from analyzed content
        
        Args:
            content_analysis: Analysis results from Content Analyzer
            num_questions: Number of questions to generate
            
        Returns:
            Quiz object with generated questions
        """
        start_time = time.time()
        
        print(f"\n{self.name} generating quiz...")
        print(f"  Target questions: {num_questions}")
        print(f"  Content topics: {len(content_analysis.main_topics)}")
        
        try:
            # Determine if we need LLM for complex generation
            use_llm = self._should_use_llm(content_analysis, num_questions)
            
            # FOLLOWING PROJECT GUIDELINES: Always use deterministic templates (0 LLM calls)
            questions = self._generate_deterministic(content_analysis, num_questions)
            
            # Create quiz object
            quiz = Quiz(
                title=f"Quiz on {content_analysis.main_topics[0] if content_analysis.main_topics else 'Educational Content'}",
                description=f"Practice quiz covering {len(content_analysis.main_topics)} main topics",
                questions=questions,
                total_questions=len(questions),
                estimated_time=max(5, len(questions) * 2),
                difficulty=content_analysis.difficulty_level
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            print(f"  Generated {len(questions)} questions")
            print(f"  Quiz difficulty: {quiz.difficulty}")
            print(f"  Estimated time: {quiz.estimated_time} minutes")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  LLM used: No (Deterministic templates per project guidelines)")
            
            return quiz
            
        except Exception as e:
            print(f"  Error generating quiz: {str(e)}")
            raise
    
    def _should_use_llm(self, content_analysis: ContentAnalysis, num_questions: int) -> bool:
        """Determine if LLM should be used based on complexity"""
        # FOLLOWING PROJECT GUIDELINES: Quiz Generator is now deterministic (0 LLM calls)
        # Always use deterministic templates regardless of complexity
        return False
    
    def _generate_deterministic(self, content_analysis: ContentAnalysis, num_questions: int) -> List[Question]:
        """Generate questions using deterministic templates (0 LLM calls)"""
        questions = []
        
        # Get available concepts and topics
        concepts = content_analysis.key_concepts[:min(num_questions, len(content_analysis.key_concepts))]
        topics = content_analysis.main_topics
        
        # Check if we're using fallback/generic content
        if not concepts or concepts[0].concept_name in ['Content Understanding', 'Knowledge Extraction', 'Educational Assessment']:
            print(f"  Using fallback content analysis - generating general educational questions")
            print(f"  These questions focus on educational concepts rather than specific document content")
            
            # Generate generic educational questions when content analysis fails
            generic_questions = [
                Question(
                    question_text="What is the primary purpose of educational content analysis?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    correct_answer="To identify key concepts and learning objectives",
                    options=[
                        "To identify key concepts and learning objectives",
                        "To reduce document file size",
                        "To change document format",
                        "To add visual elements"
                    ],
                    explanation="Content analysis helps identify what students should learn from the material.",
                    difficulty="medium",
                    topic="Educational Content"
                ),
                Question(
                    question_text="True or False: Multiple choice questions can test both knowledge and understanding.",
                    question_type=QuestionType.TRUE_FALSE,
                    correct_answer="True",
                    explanation="Well-designed multiple choice questions can assess various levels of learning.",
                    difficulty="beginner",
                    topic="Assessment Methods"
                ),
                Question(
                    question_text="Which of the following is NOT a common question type in educational assessments?",
                    question_type=QuestionType.MULTIPLE_CHOICE,
                    correct_answer="Color preference questions",
                    options=[
                        "Multiple choice questions",
                        "True/False questions",
                        "Short answer questions",
                        "Color preference questions"
                    ],
                    explanation="Color preferences are not relevant to testing educational content knowledge.",
                    difficulty="beginner",
                    topic="Question Types"
                )
            ]
            
            # Return appropriate number of generic questions
            return generic_questions[:num_questions]
        
        # Generate questions for each concept when content analysis succeeds
        for i, concept in enumerate(concepts):
            if i >= num_questions:
                break
            
            # Create multiple choice question
            question = Question(
                question_text=f"What is {concept.concept_name}?",
                question_type=QuestionType.MULTIPLE_CHOICE,
                correct_answer=concept.description[:100] + "..." if len(concept.description) > 100 else concept.description,
                options=[
                    concept.description[:100] + "..." if len(concept.description) > 100 else concept.description,
                    f"Related to {concept.related_concepts[0] if concept.related_concepts else 'general topic'}",
                    "None of the above",
                    "This concept is not defined"
                ],
                explanation=f"{concept.concept_name} is a key concept in this topic.",
                difficulty=content_analysis.difficulty_level,
                topic=topics[0] if topics else "General"
            )
            questions.append(question)
        
        # Fill remaining questions with topic-based questions
        remaining = num_questions - len(questions)
        for i in range(remaining):
            if i < len(topics):
                topic = topics[i]
                question = Question(
                    question_text=f"Which of the following is related to {topic}?",
                    question_type=QuestionType.TRUE_FALSE,
                    correct_answer="True",
                    explanation=f"{topic} is one of the main topics covered in this content.",
                    difficulty=content_analysis.difficulty_level,
                    topic=topic
                )
                questions.append(question)
        
        return questions
   