"""
Base Agent Class for the Educational Content System
All agents inherit from this class to ensure consistent interface and LLM tracking
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, uses_llm: bool = False):
        self.name = name
        self.uses_llm = uses_llm
        self.llm_calls_made = 0
        self.total_processing_time = 0.0
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return results"""
        pass
    
    def track_llm_call(self):
        """Track when this agent makes an LLM call"""
        if self.uses_llm:
            self.llm_calls_made += 1
            print(f"  → {self.name} made LLM call #{self.llm_calls_made}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics for monitoring"""
        return {
            "name": self.name,
            "uses_llm": self.uses_llm,
            "llm_calls_made": self.llm_calls_made,
            "total_processing_time": self.total_processing_time
        }
    
    def __str__(self) -> str:
        return f"{self.name} (LLM: {'Yes' if self.uses_llm else 'No'})"
