"""
File utility functions for the Educational Content System
"""

import os
import mimetypes
from typing import List, Tuple


def validate_file(file_path: str, allowed_extensions: List[str] = None) -> Tuple[bool, str]:
    """
    Validate if a file exists and has the correct format
    
    Args:
        file_path: Path to the file
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if allowed_extensions is None:
        allowed_extensions = ['.pdf']
    
    # Check if file exists
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    # Check if it's a file (not directory)
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(file_path)
    max_size = 50 * 1024 * 1024  # 50MB
    if file_size > max_size:
        return False, f"File too large: {file_size / (1024*1024):.1f}MB. Max: 50MB"
    
    # Check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1024)  # Try to read first 1KB
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"
    
    return True, ""


def get_file_info(file_path: str) -> dict:
    """
    Get basic information about a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        stat = os.stat(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        return {
            'name': os.path.basename(file_path),
            'path': os.path.abspath(file_path),
            'size': stat.st_size,
            'extension': file_ext,
            'mime_type': mimetypes.guess_type(file_path)[0] or 'unknown',
            'created': stat.st_ctime,
            'modified': stat.st_mtime
        }
    except Exception as e:
        return {
            'error': str(e)
        }


def create_output_filename(input_file: str, suffix: str = "_results", extension: str = ".txt") -> str:
    """
    Create an output filename based on input file
    
    Args:
        input_file: Input file path
        suffix: Suffix to add before extension
        extension: Output file extension
        
    Returns:
        Output filename
    """
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    return f"{base_name}{suffix}{extension}"


def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception:
        return False


def list_supported_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    List all supported files in a directory
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = ['.pdf']
    
    supported_files = []
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in extensions:
                    supported_files.append(file_path)
    except Exception:
        pass
    
    return supported_files
