"""
Text chunking utilities for RAG system
Handles document splitting into manageable chunks for vector storage
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    chunk_id: str
    start_char: int
    end_char: int
    document_id: str
    chunk_index: int
    metadata: Dict[str, any]


class TextChunker:
    """Handles text chunking for vector storage"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum acceptable chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position for this chunk
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                end = self._find_sentence_boundary(text, start, end)
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Skip chunks that are too small (unless it's the last one)
            if len(chunk_text) < self.min_chunk_size and end < len(text):
                start = end
                continue
            
            # Create chunk object
            chunk = TextChunk(
                text=chunk_text,
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                start_char=start,
                end_char=end,
                document_id=document_id,
                chunk_index=chunk_index,
                metadata={
                    "length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "has_content": bool(chunk_text.strip())
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(start + 1, end - self.chunk_overlap)
            chunk_index += 1
            
            # Safety check to prevent infinite loops
            if start >= len(text):
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _find_sentence_boundary(self, text: str, start: int, target_end: int) -> int:
        """
        Find a good sentence boundary near the target end position
        
        Args:
            text: Full text
            start: Start of current chunk
            target_end: Target end position
            
        Returns:
            Actual end position at sentence boundary
        """
        # Look for sentence endings within a reasonable range
        search_start = max(start, target_end - 200)
        search_end = min(len(text), target_end + 200)
        
        # Common sentence endings
        sentence_endings = ['.', '!', '?', '\n\n']
        
        # Look backwards from target_end for sentence boundary
        for i in range(target_end, search_start, -1):
            if text[i] in sentence_endings:
                # Found a sentence boundary
                return i + 1
        
        # If no sentence boundary found, use target_end
        return target_end
    
    def chunk_by_sections(self, text: str, document_id: str) -> List[TextChunk]:
        """
        Alternative chunking method that respects document sections
        
        Args:
            text: Input text to chunk
            document_id: Unique identifier for the document
            
        Returns:
            List of TextChunk objects
        """
        # Split by double newlines (common section separators)
        sections = text.split('\n\n')
        
        chunks = []
        chunk_index = 0
        current_pos = 0
        
        for section in sections:
            if not section.strip():
                current_pos += len(section) + 2  # +2 for \n\n
                continue
            
            # If section is too long, chunk it further
            if len(section) > self.chunk_size:
                section_chunks = self.chunk_text(section, document_id)
                for chunk in section_chunks:
                    # Adjust positions to account for full document
                    chunk.start_char += current_pos
                    chunk.end_char += current_pos
                    chunk.chunk_index = chunk_index
                    chunk.chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                # Section fits in one chunk
                chunk = TextChunk(
                    text=section.strip(),
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    start_char=current_pos,
                    end_char=current_pos + len(section),
                    document_id=document_id,
                    chunk_index=chunk_index,
                    metadata={
                        "length": len(section),
                        "word_count": len(section.split()),
                        "section": True,
                        "has_content": bool(section.strip())
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            current_pos += len(section) + 2  # +2 for \n\n
        
        return chunks
    
    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, any]:
        """Get statistics about the chunking process"""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        total_text_length = sum(len(chunk.text) for chunk in chunks)
        avg_chunk_size = total_text_length / total_chunks
        
        return {
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "average_chunk_size": avg_chunk_size,
            "min_chunk_size": min(len(chunk.text) for chunk in chunks),
            "max_chunk_size": max(len(chunk.text) for chunk in chunks),
            "chunk_size_variance": sum((len(chunk.text) - avg_chunk_size) ** 2 for chunk in chunks) / total_chunks
        }
