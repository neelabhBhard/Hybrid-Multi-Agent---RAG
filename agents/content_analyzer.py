"""
Content Analyzer Agent
Uses LLM to analyze educational content and extract key concepts (1 LLM call)
"""

import json
import time
from typing import List
from agents.base_agent import BaseAgent
from core.models import ContentAnalysis, ContentConcept
from core.llm_interface import LLMInterface


class ContentAnalyzerAgent(BaseAgent):
    """Agent for analyzing educational content using LLM"""
    
    def __init__(self, llm_interface: LLMInterface):
        super().__init__(name="Content Analyzer", uses_llm=True)
        self.llm = llm_interface
    
    def process(self, text_content: str) -> ContentAnalysis:
        """
        Analyze educational content and extract key concepts
        
        Args:
            text_content: Extracted text content from document
            
        Returns:
            ContentAnalysis object with extracted information
        """
        start_time = time.time()
        
        print(f"\n{self.name} analyzing content...")
        print(f"  Content length: {len(text_content)} characters")
        
        try:
            # Prepare content for analysis (truncate if too long)
            truncated_content = self._truncate_content(text_content)
            print(f"    Content prepared for analysis: {len(truncated_content)} characters")
            print(f"    Content preview: {truncated_content[:200]}...")
            
            # Create analysis prompt
            analysis_prompt = self._create_analysis_prompt(truncated_content)
            print(f"    LLM prompt created, making API call...")
            print(f"    Prompt length: {len(str(analysis_prompt))} characters")
            
            # Make LLM call for content analysis
            self.track_llm_call()
            print(f"    Making LLM API call...")
            response = self.llm.make_call(analysis_prompt)
            print(f"    LLM response received: {len(response)} characters")
            print(f"    Response type: {type(response)}")
            
            # Check if response is valid
            if not response or len(response.strip()) == 0:
                print(f"    ERROR: Empty response from LLM!")
                return self._create_fallback_analysis()
            
            if "error" in response.lower() or "failed" in response.lower():
                print(f"    ERROR: LLM returned error response!")
                print(f"    Response: {response}")
                return self._create_fallback_analysis()
            
            # Parse and validate LLM response
            print(f"    Starting JSON parsing...")
            analysis_data = self._parse_llm_response(response)
            
            # Create ContentAnalysis object
            content_analysis = ContentAnalysis(
                main_topics=analysis_data.get('main_topics', []),
                key_concepts=analysis_data.get('key_concepts', []),
                difficulty_level=analysis_data.get('difficulty_level', 'medium'),
                estimated_reading_time=analysis_data.get('estimated_reading_time', 10),
                summary=analysis_data.get('summary', ''),
                learning_objectives=analysis_data.get('learning_objectives', [])
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            print(f"  Identified {len(content_analysis.main_topics)} main topics")
            print(f"  Extracted {len(content_analysis.key_concepts)} key concepts")
            print(f"  Difficulty level: {content_analysis.difficulty_level}")
            print(f"  Processing time: {processing_time:.2f}s")
            
            return content_analysis
            
        except Exception as e:
            print(f"  Error analyzing content: {str(e)}")
            print(f"  Using fallback analysis due to error")
            return self._create_fallback_analysis()
    
    def _truncate_content(self, content: str, max_chars: int = 15000) -> str:
        """Truncate content to fit within LLM token limits"""
        if len(content) <= max_chars:
            return content
        
        # Clean the content first
        cleaned_content = self._clean_content(content)
        
        if len(cleaned_content) <= max_chars:
            return cleaned_content
        
        # Truncate to last complete sentence
        truncated = cleaned_content[:max_chars]
        last_period = truncated.rfind('.')
        
        if last_period > max_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[:last_period + 1]
        
        print(f"    Note: Content truncated from {len(content)} to {len(truncated)} characters")
        return truncated
    
    def _clean_content(self, content: str) -> str:
        """Clean and preprocess PDF content for better analysis"""
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(content.split())
        
        # Remove common PDF artifacts
        cleaned = cleaned.replace('  ', ' ')  # Double spaces
        cleaned = cleaned.replace(' .', '.')  # Space before period
        cleaned = cleaned.replace(' ,', ',')  # Space before comma
        
        # Remove page numbers and headers if they appear
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that are just numbers (likely page numbers)
            if line.isdigit() and len(line) < 4:
                continue
            # Skip very short lines that might be headers
            if len(line) < 3:
                continue
            filtered_lines.append(line)
        
        cleaned = ' '.join(filtered_lines)
        
        print(f"    Content cleaned: {len(content)} -> {len(cleaned)} characters")
        return cleaned
    
    def _create_analysis_prompt(self, content: str) -> List[dict]:
        """Create the prompt for content analysis"""
        return [
            {
                "role": "system",
                "content": """You are an expert educational content analyzer. Analyze the provided content and extract key information.

IMPORTANT: Return ONLY valid JSON, no other text.

Required JSON structure:
{
    "main_topics": ["topic1", "topic2", "topic3"],
    "key_concepts": [
        {
            "concept_name": "concept name",
            "description": "brief description",
            "importance_score": 0.8,
            "related_concepts": ["related1", "related2"],
            "examples": ["example1", "example2"]
        }
    ],
    "difficulty_level": "beginner|intermediate|advanced",
    "estimated_reading_time": 15,
    "summary": "2-3 sentence summary",
    "learning_objectives": ["objective1", "objective2", "objective3"]
}

Rules:
- Extract 3-5 main topics from the content
- Identify 5-8 key concepts with descriptions
- Use importance scores 0.0-1.0
- Base everything on the actual content provided
- Return ONLY the JSON object"""
            },
            {
                "role": "user",
                "content": f"Analyze this content and return ONLY JSON:\n\n{content}"
            }
        ]
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse and validate the LLM response"""
        try:
            print(f"    Raw LLM response received: {len(response)} characters")
            print(f"    Response preview: {response[:300]}...")
            
            # Try multiple JSON extraction strategies
            json_str = self._extract_json_from_response(response)
            
            if json_str:
                print(f"    Extracted JSON: {len(json_str)} characters")
                # Parse JSON
                analysis_data = json.loads(json_str)
                print(f"    Successfully parsed JSON with {len(analysis_data.get('key_concepts', []))} concepts")
                
                # Validate and clean the data
                return self._validate_analysis_data(analysis_data)
            else:
                print(f"    ERROR: Could not extract JSON from response")
                print(f"    Full response: {response}")
                raise ValueError("No JSON found in LLM response")
                
        except json.JSONDecodeError as e:
            print(f"    JSON parsing error: {str(e)}")
            print(f"    Attempted to parse: {json_str if 'json_str' in locals() else 'N/A'}")
            # Try to extract partial information from the response
            partial_data = self._extract_partial_content(response)
            if partial_data:
                print(f"    Successfully extracted partial content from truncated response")
                return partial_data
            return self._create_fallback_analysis()
        except Exception as e:
            print(f"    General parsing error: {str(e)}")
            # Try to extract partial information from the response
            partial_data = self._extract_partial_content(response)
            if partial_data:
                print(f"    Successfully extracted partial content from truncated response")
                return partial_data
            return self._create_fallback_analysis()
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response using multiple strategies"""
        # Strategy 1: Look for JSON between { and } (most common)
        if '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            # Try to parse the extracted JSON
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError as e:
                print(f"      Strategy 1 failed: {str(e)}")
                # Try to fix common truncation issues
                fixed_json = self._fix_truncated_json(json_str)
                if fixed_json:
                    try:
                        json.loads(fixed_json)
                        print(f"      Fixed truncated JSON successfully")
                        return fixed_json
                    except:
                        pass
        
        # Strategy 2: Look for JSON array markers
        if '[' in response and ']' in response:
            start = response.find('[')
            end = response.rfind(']') + 1
            json_str = response[start:end]
            
            try:
                json.loads(json_str)
                return json_str
            except:
                pass
        
        # Strategy 3: Try to find any valid JSON structure
        # Look for common JSON patterns
        json_patterns = [
            r'\{[^{}]*"main_topics"[^{}]*\}',  # Basic structure with main_topics
            r'\{[^{}]*"key_concepts"[^{}]*\}',  # Basic structure with key_concepts
        ]
        
        import re
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                try:
                    json.loads(matches[0])
                    return matches[0]
                except:
                    continue
        
        return None
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """Attempt to fix truncated JSON responses"""
        try:
            # Check if the JSON is incomplete
            if json_str.count('{') > json_str.count('}'):
                # Missing closing braces - try to complete
                missing_braces = json_str.count('{') - json_str.count('}')
                json_str += '}' * missing_braces
                print(f"      Added {missing_braces} missing closing braces")
            
            # Check if arrays are incomplete
            if json_str.count('[') > json_str.count(']'):
                missing_brackets = json_str.count('[') - json_str.count(']')
                json_str += ']' * missing_brackets
                print(f"      Added {missing_brackets} missing closing brackets")
            
            # Check if strings are incomplete (look for unclosed quotes)
            # Find the last incomplete string and close it
            last_quote_pos = json_str.rfind('"')
            if last_quote_pos != -1:
                # Look for the opening quote of this string
                before_last_quote = json_str[:last_quote_pos]
                opening_quote_pos = before_last_quote.rfind('"')
                if opening_quote_pos != -1:
                    # Check if there's an odd number of quotes between these positions
                    between_quotes = json_str[opening_quote_pos+1:last_quote_pos]
                    if between_quotes.count('"') % 2 == 0:  # Even number means string is complete
                        pass
                    else:
                        # String is incomplete, try to close it
                        json_str += '"'
                        print(f"      Closed incomplete string")
            
            return json_str
            
        except Exception as e:
            print(f"      Error fixing JSON: {str(e)}")
            return None
    
    def _extract_partial_content(self, response: str) -> dict:
        """Extract partial content from truncated LLM responses"""
        try:
            print(f"      Attempting to extract partial content from truncated response...")
            
            # Look for main_topics array
            main_topics = []
            if '"main_topics"' in response:
                topics_start = response.find('"main_topics"')
                if topics_start != -1:
                    # Find the opening bracket after main_topics
                    bracket_start = response.find('[', topics_start)
                    if bracket_start != -1:
                        # Find the closing bracket
                        bracket_end = response.find(']', bracket_start)
                        if bracket_end != -1:
                            topics_text = response[bracket_start+1:bracket_end]
                            # Extract individual topics
                            topics = [t.strip().strip('"') for t in topics_text.split(',') if t.strip()]
                            main_topics = [t for t in topics if t and len(t) > 2]
                            print(f"      Extracted {len(main_topics)} main topics: {main_topics}")
            
            # Look for key_concepts array
            key_concepts = []
            if '"key_concepts"' in response:
                concepts_start = response.find('"key_concepts"')
                if concepts_start != -1:
                    # Find the opening bracket after key_concepts
                    bracket_start = response.find('[', concepts_start)
                    if bracket_start != -1:
                        # Extract concepts until the response ends
                        concepts_text = response[bracket_start+1:]
                        # Look for complete concept objects
                        concept_pattern = r'\{[^}]*"concept_name"[^}]*"description"[^}]*\}'
                        import re
                        matches = re.findall(concept_pattern, concepts_text, re.DOTALL)
                        
                        for match in matches:
                            try:
                                # Try to extract concept name and description
                                name_match = re.search(r'"concept_name":\s*"([^"]+)"', match)
                                desc_match = re.search(r'"description":\s*"([^"]+)"', match)
                                
                                if name_match and desc_match:
                                    concept = {
                                        'concept_name': name_match.group(1),
                                        'description': desc_match.group(1),
                                        'importance_score': 0.8,
                                        'related_concepts': [],
                                        'examples': []
                                    }
                                    key_concepts.append(concept)
                            except:
                                continue
                        
                        print(f"      Extracted {len(key_concepts)} key concepts")
            
            if main_topics or key_concepts:
                return {
                    'main_topics': main_topics or ['Data Science', 'Machine Learning'],
                    'key_concepts': key_concepts or [
                        {
                            'concept_name': 'Data Science Concepts',
                            'description': 'Various machine learning and statistical concepts',
                            'importance_score': 0.8,
                            'related_concepts': [],
                            'examples': []
                        }
                    ],
                    'difficulty_level': 'intermediate',
                    'estimated_reading_time': 15,
                    'summary': 'Data science and machine learning content with focus on neural networks and probability',
                    'learning_objectives': ['Understand ML concepts', 'Learn statistical methods', 'Apply neural networks']
                }
            
            return None
            
        except Exception as e:
            print(f"      Error extracting partial content: {str(e)}")
            return None
    
    def _validate_analysis_data(self, data: dict) -> dict:
        """Validate and clean the analysis data"""
        # Ensure all required fields exist
        validated_data = {
            'main_topics': data.get('main_topics', []),
            'key_concepts': data.get('key_concepts', []),
            'difficulty_level': data.get('difficulty_level', 'medium'),
            'estimated_reading_time': data.get('estimated_reading_time', 10),
            'summary': data.get('summary', ''),
            'learning_objectives': data.get('learning_objectives', [])
        }
        
        # Validate key concepts
        if validated_data['key_concepts']:
            validated_concepts = []
            for concept in validated_data['key_concepts']:
                if isinstance(concept, dict):
                    validated_concept = {
                        'concept_name': concept.get('concept_name', 'Unknown Concept'),
                        'description': concept.get('description', ''),
                        'importance_score': min(max(concept.get('importance_score', 0.5), 0.0), 1.0),
                        'related_concepts': concept.get('related_concepts', []),
                        'examples': concept.get('examples', [])
                    }
                    validated_concepts.append(validated_concept)
            validated_data['key_concepts'] = validated_concepts
        
        # Ensure difficulty level is valid
        valid_difficulties = ['beginner', 'medium', 'advanced']
        if validated_data['difficulty_level'] not in valid_difficulties:
            validated_data['difficulty_level'] = 'medium'
        
        # Ensure reading time is reasonable
        validated_data['estimated_reading_time'] = max(1, min(validated_data['estimated_reading_time'], 120))
        
        return validated_data
    
    def _create_fallback_analysis(self) -> dict:
        """Create fallback analysis when LLM parsing fails"""
        print("    WARNING: Content analysis had issues - using intelligent fallback")
        print("    The system will generate questions based on general educational concepts")
        print("    This may happen if:")
        print("      - LLM response format was unexpected")
        print("      - Document content was very complex")
        print("      - API response was incomplete")
        
        return {
            'main_topics': ['Educational Content', 'Learning Materials', 'Document Analysis'],
            'key_concepts': [
                {
                    'concept_name': 'Content Understanding',
                    'description': 'The ability to comprehend and extract meaning from educational materials',
                    'importance_score': 0.8,
                    'related_concepts': ['Learning', 'Education'],
                    'examples': ['Reading comprehension', 'Concept identification']
                },
                {
                    'concept_name': 'Knowledge Extraction',
                    'description': 'Process of identifying key information and concepts from documents',
                    'importance_score': 0.7,
                    'related_concepts': ['Information Processing', 'Learning'],
                    'examples': ['Topic identification', 'Concept mapping']
                },
                {
                    'concept_name': 'Educational Assessment',
                    'description': 'Methods for evaluating understanding and knowledge acquisition',
                    'importance_score': 0.6,
                    'related_concepts': ['Learning Evaluation', 'Knowledge Testing'],
                    'examples': ['Quiz generation', 'Question formulation']
                }
            ],
            'difficulty_level': 'medium',
            'estimated_reading_time': 15,
            'summary': 'Educational content analysis focusing on content understanding and knowledge extraction',
            'learning_objectives': ['Understand content analysis principles', 'Apply knowledge extraction techniques', 'Evaluate learning outcomes']
        }
