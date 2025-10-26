"""
Document Processor Agent
Extracts text from PDF documents without using LLM calls (deterministic agent)
"""

import os
import time
from typing import Optional
import PyPDF2
from agents.base_agent import BaseAgent
from core.models import DocumentInfo


class DocumentProcessorAgent(BaseAgent):
    """Agent for processing PDF documents and extracting text"""
    
    def __init__(self):
        super().__init__(name="Document Processor", uses_llm=False)
        self.supported_formats = ['.pdf', '.txt']
    
    def process(self, file_path: str) -> DocumentInfo:
        """
        Process a PDF document and extract text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentInfo object with extracted information
        """
        start_time = time.time()
        
        print(f"\n {self.name} processing: {os.path.basename(file_path)}")
        
        try:
            # Validate file
            if not self._is_valid_file(file_path):
                raise ValueError(f"Unsupported file format. Supported: {self.supported_formats}")
            
            # Extract document information
            file_info = self._get_file_info(file_path)
            
            # Extract text from document
            if file_path.lower().endswith('.pdf'):
                extracted_text = self._extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.txt'):
                extracted_text = self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Create document info
            document_info = DocumentInfo(
                filename=os.path.basename(file_path),
                file_size=file_info['size'],
                pages=file_info['pages'],
                text_length=len(extracted_text),
                processing_time=processing_time
            )
            
            print(f"  ✓ Successfully processed {file_info['pages']} pages")
            print(f"  ✓ Extracted {len(extracted_text)} characters")
            print(f"  ✓ Processing time: {processing_time:.2f}s")
            
            # Store extracted text for other agents to use
            self._extracted_text = extracted_text
            
            return document_info
            
        except Exception as e:
            print(f"  Error processing document: {str(e)}")
            raise
    
    def get_extracted_text(self) -> str:
        """Get the extracted text from the last processed document"""
        return getattr(self, '_extracted_text', '')
    
    def _is_valid_file(self, file_path: str) -> bool:
        """Check if file format is supported"""
        if not os.path.exists(file_path):
            return False
        
        file_ext = os.path.splitext(file_path)[1].lower()
        return file_ext in self.supported_formats
    
    def _get_file_info(self, file_path: str) -> dict:
        """Get basic file information"""
        file_stat = os.stat(file_path)
        
        if file_path.lower().endswith('.pdf'):
            # Count pages in PDF
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
        elif file_path.lower().endswith('.txt'):
            # For text files, estimate pages (assuming ~2000 characters per page)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                page_count = max(1, len(content) // 2000)
        else:
            page_count = 1
        
        return {
            'size': file_stat.st_size,
            'pages': page_count
        }
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file"""
        extracted_text = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += f"\n--- Page {page_num + 1} ---\n"
                        extracted_text += page_text
                        extracted_text += "\n"
                except Exception as e:
                    print(f"    Warning: Could not extract text from page {page_num + 1}: {str(e)}")
                    continue
        
        # Clean up the extracted text
        extracted_text = self._clean_text(extracted_text)
        
        return extracted_text
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text content from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Clean up the extracted text
            cleaned_content = self._clean_text(content)
            
            return cleaned_content
            
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                cleaned_content = self._clean_text(content)
                return cleaned_content
            except Exception as e:
                raise ValueError(f"Could not read text file with any encoding: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip page markers
            if line.startswith('--- Page') and line.endswith('---'):
                continue
            
            cleaned_lines.append(line)
        
        # Join lines with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive spaces
        import re
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def get_stats(self) -> dict:
        """Get agent statistics including text processing metrics"""
        stats = super().get_stats()
        stats.update({
            "extracted_text_length": len(getattr(self, '_extracted_text', '')),
            "supported_formats": self.supported_formats
        })
        return stats
